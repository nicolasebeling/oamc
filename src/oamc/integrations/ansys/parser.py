"""Contains the APDLParser class as well as related enums, dataclasses,
and utility functions.
"""

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from functools import cached_property
from pathlib import Path
from time import perf_counter as clock
from typing import Any, Callable, Iterator

import numpy
from numpy.typing import NDArray

from oamc.constants import NODE_COUNT_FROM_ELEMENT_TYPE, SOLID_ELEMENT_TYPES, SURFACE_ELEMENT_TYPES
from oamc.enums import Direction, ElementType
from oamc.fem.bc import BC
from oamc.fem.material import IsotropicMaterial, OrthotropicMaterial
from oamc.fem.mesh import Mesh, SolidMesh, SurfaceMesh
from oamc.fem.model import SolidModel

logger = logging.getLogger(__name__)


# NOTE: The number of entities and the number of fields per line given
# in EBLOCK and NBLOCK commands and the subsequent format specifier are
# often incorrect when exported from Ansys Mechanical. Do not rely on
# these numbers alone; rather read the blocks line-by-line.


class APDLEntityType(Enum):
    NODE = "NODE"
    ELEMENT = "ELEMENT"
    KEYPOINT = "KEYPOINT"
    LINE = "LINE"
    AREA = "AREA"
    VOLUME = "VOLUME"


class APDLProcessor(Enum):
    """APDL processors (routines).

    Values correspond to internal codes [1]_.

    References
    ----------
    .. [1] https://mapdl.docs.pyansys.com/version/stable/mapdl_commands/apdl/_autosummary/ansys.mapdl.core._commands.apdl.parameter_definition.ParameterDefinition.get.html
    """

    BEGIN = 0
    PREP7 = 17
    SOLUTION = 21
    POST1 = 31
    POST26 = 36
    AUX2 = 52
    AUX3 = 53
    AUX12 = 62
    AUX15 = 65


@dataclass(slots=True)
class APDLComponent:
    """
    Parameters
    ----------
    type : oamc.integrations.ansys.parser.APDLEntityType
        Type of entities in the component.
    entities : list of int
        Entity numbers belonging to the component.
    """

    type: APDLEntityType
    entities: list[int]


@dataclass(slots=True)
class APDLElement:
    """
    Parameters
    ----------
    material : int
        Material number.
    type : int
        Element type number.
    real_constant_set : int
        Real constant set number.
    section : int
        Section number.
    cosy : int
        Element coordinate system number.
    connectivity : list of int
        Ordered list of node numbers.
    """

    material: int
    type: int
    real_constant_set: int
    section: int
    cosy: int
    connectivity: NDArray


@dataclass(slots=True)
class APDLState:
    # Current processor:
    processor: APDLProcessor = APDLProcessor.BEGIN

    # Model metadata:
    title: str | None = None
    units: str | None = None

    # Geometric entities:
    cosys: dict[int, dict[str, float]] = field(default_factory=dict)
    nodes: dict[int, tuple[float, float, float]] = field(default_factory=dict)
    elements: dict[int, APDLElement] = field(default_factory=dict)
    element_types: dict[int, dict[str, Any]] = field(default_factory=dict)
    real_constant_sets: dict[int, list[int]] = field(default_factory=dict)
    materials: dict[int, dict[str, float]] = field(default_factory=dict)
    sections: dict[int, dict[str, float]] = field(default_factory=dict)

    # Current variables:
    variables: dict[str, float] = field(default_factory=dict)

    # Components (named selections):
    components: dict[str, APDLComponent] = field(default_factory=dict)

    # Current selections:
    selected_keypoints: set[int] = field(default_factory=set)
    selected_lines: set[int] = field(default_factory=set)
    selected_areas: set[int] = field(default_factory=set)
    selected_volumes: set[int] = field(default_factory=set)
    selected_nodes: set[int] = field(default_factory=set)
    selected_elements: set[int] = field(default_factory=set)
    selected_real_constant_sets: set[int] = field(default_factory=set)
    selected_components: set[str] = field(default_factory=set)

    # Attribute pointers keep track of currently active attributes:
    attribute_pointers: dict[str, int | None] = field(
        default_factory=lambda: {
            # Material:
            "MAT": None,
            # Local element type:
            "TYPE": None,
            # Real constant set:
            "REAL": None,
            # Section number:
            "SECNUM": None,
            # Element coordinate system:
            "ESYS": 0,
            # Working coordinate system:
            "CSYS": 0,
            # Result coordinate system:
            "RSYS": 0,
        }
    )

    # Last used indices are required for automatic index assignment:
    last_indices: dict[APDLEntityType, int] = field(
        default_factory=lambda: {
            APDLEntityType.NODE: 0,
            APDLEntityType.ELEMENT: 0,
            APDLEntityType.KEYPOINT: 0,
            APDLEntityType.LINE: 0,
            APDLEntityType.AREA: 0,
            APDLEntityType.VOLUME: 0,
        }
    )

    sfcontrol: dict[str, int] = field(default_factory=dict)

    dbc: list[BC] = field(default_factory=list)
    nbc: list[BC] = field(default_factory=list)


def tokenize(string: str, sep: str | None = None) -> list[str]:
    """
    Tokenize the given string, remove leading and trailing whitespace
    from the tokens.

    Parameters
    ----------
    string : str
        String to tokenize.
    sep : str, default: None (any whitespace)
        Separator used to split the string.

    Return
    ------
    list of str
        List of tokens without leading and trailing whitespace.

    Examples
    --------
    >>> tokenize("et, 2, 154 ", ",")
    ["et", "2", "154"]
    """
    return [token.strip() for token in string.split(sep)]


def parse_line(line: str) -> tuple[str, list[str]]:
    """Split an APDL line into command and arguments.

    Removes comments and whitespace. Capitalizes commands. Quoted
    arguments are returned as a single string, regardless of commas
    inside the quotes.

    Parameters
    ----------
    line : str
        Raw APDL line.

    Returns
    -------
    str
        Command.
    list of str
        List of arguments.
    """

    # Strip comments and whitespace:
    line = re.sub(
        pattern=r"(!|/com).*",
        repl="",
        string=line,
        flags=re.IGNORECASE,
    ).strip()

    # If the line is empty:
    if not line:
        return "", []

    tokens = tokenize(line, ",")

    return tokens[0].upper(), tokens[1:]

    # command, *args_str = line.split(sep=",", maxsplit=1)
    # command = command.strip().upper()
    # args_str = args_str[0] if args_str else ""

    # # Split args by commas except in quotes:
    # args = []
    # char_buffer = []
    # quote_char = None
    # for char in args_str:
    #     if char in "'\"" and quote_char is None:
    #         quote_char = char
    #     elif char == quote_char:
    #         quote_char = None
    #     elif char == "," and quote_char is None:
    #         args.append("".join(char_buffer).strip())
    #         char_buffer = []
    #     else:
    #         char_buffer.append(char)
    # if char_buffer:
    #     args.append("".join(char_buffer).strip())

    # return command, args


def fields_per_line(format_line: str) -> int:
    """Determine the number of fields per line from an APDL format line.

    Parameters
    ----------
    format_line : str
        APDL format line.

    Returns
    -------
    int
        Number of fields per line in the following block.

    Examples
    --------
    >>> fields_per_line("(1i9,3e20.9e3)")
    4

    >>> fields_per_line("(i9,3e20.9e3)")
    4

    >>> fields_per_line("(10x,3i8)")
    3

    >>> fields_per_line("(2i10,2f12.6,1e20.9e3)")
    5

    >>> fields_per_line("(5i10)")
    5
    """

    # Remove leading/trailing parentheses and spaces, convert to lowercase:
    format_line = format_line.lstrip(" (").rstrip(") ")

    tokens = tokenize(format_line, ",")

    fields_per_line = 0

    for token in tokens:
        # Skip blank tokens:
        if not token:
            continue

        # Match optional leading number + type character:
        re_match = re.match(r"(\d*)([a-z])", token)
        if not re_match:
            # Skip invalid format strings:
            continue

        count, type = re_match.groups()
        count = int(count) if count else 1

        fields_per_line += count

    return fields_per_line


_HANDLER_REGISTRY: dict[str, tuple[Callable, bool]] = {}


def command(*names: str, needs_iterator: bool = False) -> Callable[[Callable], Callable]:
    def decorator(handler: Callable) -> Callable:
        for name in names:
            _HANDLER_REGISTRY[name.upper()] = (handler, needs_iterator)
        return handler

    return decorator


class APDLParser:
    """Parses an APDL file.

    Attributes
    ----------
    path : pathlib.Path
        Path to the APDL file.
    state : APDLState
        Current state of the APDL parser.

    Examples
    --------
    ```
    parser = APDLParser("C:/path/to/file.dat")
    model = parser.get_solid_model()
    ```
    """

    def __init__(self, path: str):
        """Initialize a new instance.

        Parameters
        ----------
        path : str
            Path to the APDL file.
        """

        self.path = Path(path).resolve()
        if not self.path.exists():
            raise FileNotFoundError(self.path)

    def __setattr__(self, name, value):
        if name == "path":
            self.__dict__.pop("state", None)
        super().__setattr__(name, value)

    @cached_property
    def state(self) -> APDLState:
        """Parse the APDL file.

        Returns
        -------
        oamc.fem.readers.APDLState
            State object containing all parsed data and the final state.
        """

        start = clock()

        self.state = APDLState()

        line_iterator = enumerate(open(self.path, "r"), start=1)

        for line_number, line in line_iterator:
            for subline in line.split("$"):
                command, args = parse_line(subline)

                # Skip empty lines:
                if not command:
                    continue

                # TODO: Handle APDL equations and star commands.

                if command in _IGNORED_COMMANDS:
                    logger.debug(f"Ignoring command {command} in line {line_number}.")
                    continue

                entry = _HANDLER_REGISTRY.get(command)

                if entry is None:
                    logger.warning(
                        f"Unknown command {command} in line {line_number}: {line.strip()}"
                    )
                    continue

                handler, needs_iterator = entry

                try:
                    if needs_iterator:
                        handler(self.state, args, line_iterator)
                    else:
                        handler(self.state, args)
                except Exception as e:
                    raise Exception(
                        f"The following exception occurred while processing "
                        f"{command} in line {line_number}: {e}"
                    ) from e

        logger.info(f"APDL file {self.path} parsed in {round(clock() - start, 3)} seconds.")

        return self.state

    def apply_displacement(
        self,
        value: float,
        direction: Direction,
        name: str,
    ) -> None:
        """Apply a displacement to the nodes of the given named
        selection (component).

        Parameters
        ----------
        value : float
            Magnitude of the displacement.
        direction : oamc.enums.Direction
            Direction of the displacement.
        name : str
            Name of the named selection (component) to which the
            displacement is applied.
        """

        start = clock()

        component = self.state.components[name]
        if component.type != APDLEntityType.NODE:
            raise ValueError("Displacements can only be applied to node components.")

        for node in component.entities:
            self.state.dbc.append(
                BC(
                    node=node,
                    direction=direction,
                    value=value,
                )
            )

        logger.info(
            f"Displacement applied to component {name} in {round(clock() - start, 3)} seconds."
        )

    def apply_bearing_load(
        self,
        force: tuple[float, float, float],
        target: str,
    ) -> None:
        """
        Apply a bearing load to the elements of the given named selection
        (component).

        Parameters
        ----------
        force : tuple of float
            Force vector acting on the target.
        target : str
            Name of the named selection / component which the force is
            applied to.

        Notes
        -----
        This function only exists because parsing bearing loads from
        APDL files is non-trivial and not yet implemented. It is
        intended as a temporary workaround and will be removed in
        future versions.
        """

        start = clock()

        component = self.state.components[target]
        if component.type != APDLEntityType.ELEMENT:
            raise ValueError("Pressures can only be applied to element components.")

        elements = [self.state.elements[element] for element in component.entities]

        force = numpy.array(force, dtype=float)
        value = numpy.linalg.norm(force)
        direction = force / value

        projected_areas = []
        inward_normals = []
        for element in elements:
            if (
                self.state.element_types[element.type]["Ename"] == ElementType.MESH200
                and len(element.connectivity) == 8
                and element.connectivity[2] == element.connectivity[3] == element.connectivity[6]
            ):
                i = numpy.array(self.state.nodes[element.connectivity[0]])
                j = numpy.array(self.state.nodes[element.connectivity[1]])
                k = numpy.array(self.state.nodes[element.connectivity[2]])
                cross = numpy.cross(j - i, k - i)
                signed_projected_area = numpy.dot(direction, cross) / 2
                if signed_projected_area < 0:
                    projected_areas.append(abs(signed_projected_area))
                else:
                    projected_areas.append(0)
                inward_normals.append(-cross / numpy.linalg.norm(cross))
            else:
                raise NotImplementedError(
                    "Surface force application is only implemented for TRI6 MESH200 elements currently."
                )

        total_projected_area = sum(projected_areas)
        pressure = value / total_projected_area if total_projected_area > 0 else 0

        for element, projected_area, inward_normal in zip(
            elements, projected_areas, inward_normals
        ):
            if (
                self.state.element_types[element.type]["Ename"] == ElementType.MESH200
                and len(element.connectivity) == 8
                and element.connectivity[2] == element.connectivity[3] == element.connectivity[6]
            ):
                if projected_area > 0:
                    midside_force = (
                        pressure / numpy.dot(direction, inward_normal) * projected_area / 3
                    )
                    for i in range(3):
                        self.state.nbc.append(
                            BC(
                                node=element.connectivity[4],
                                direction=Direction(i),
                                value=midside_force * inward_normal[i],
                            )
                        )
                        self.state.nbc.append(
                            BC(
                                node=element.connectivity[5],
                                direction=Direction(i),
                                value=midside_force * inward_normal[i],
                            )
                        )
                        self.state.nbc.append(
                            BC(
                                node=element.connectivity[7],
                                direction=Direction(i),
                                value=midside_force * inward_normal[i],
                            )
                        )
            else:
                raise NotImplementedError(
                    "Surface force application is only implemented for TRI6 MESH200 elements currently."
                )

        logger.info(
            f"Bearing load applied to component {target} in {round(clock() - start, 3)} seconds."
        )

    def get_named_selection(self, name: str) -> list[int]:
        """Get the entity numbers of a named selection (component).

        Parameters
        ----------
        name : str
            Name of the component.

        Returns
        -------
        list of int
            List of entity numbers in the component. These numbers are
            not the same as the entity indices in the returned mesh.
            They should only be used in combination with the mapping
            from Ansys entity numbers to OAMC entity indices.

        Raises
        ------
        ValueError
            If no component with the given name exists.
        """
        try:
            component = self.state.components[name]
        except KeyError as e:
            raise ValueError(f'Named selection (component) "{name}" does not exist.') from e

        return component.entities

    def get_nodes(self) -> tuple[NDArray, dict[int, int]]:
        nodes = numpy.empty((len(self.state.nodes), 3), dtype=float)
        node_number_to_index = {}
        for index, (number, coordinates) in enumerate(self.state.nodes.items()):
            nodes[index] = coordinates
            node_number_to_index[number] = index
        return nodes, node_number_to_index

    def get_surface_mesh(self, name: str) -> tuple[Mesh, dict[int, int], dict[int, int]]:
        """Translate a named selection (component) of elements to an
        OAMC surface mesh.

        Parameters
        ----------
        name : str
            Name of the component.

        Returns
        -------
        oamc.fem.SurfaceMesh
            Translated OAMC surface mesh.
        dict of int to int
            Mapping from original node numbers to new node indices.
        dict of int to int
            Mapping from original element numbers to new element indices.

        Raises
        ------
        ValueError
            If
            - the component does not exist,
            - the component does not contain elements,
            - the component does not contain surface elements,
            - the component contains elements of different types.
        """
        try:
            component = self.state.components[name]
        except KeyError as e:
            raise ValueError(f"Named selection (component) '{name}' does not exist.") from e

        if component.type != APDLEntityType.ELEMENT:
            raise ValueError(
                f"Surface meshes can only be generated from ELEMENT selections but "
                f"{name} contains entities of type {component.type}."
            )

        nodes, node_number_to_index = self.get_nodes()

        type = None
        connectivity = []
        elem_number_to_index: dict[int, int] = {}
        for element_index, element_number in enumerate(component.entities):
            element = self.state.elements[element_number]
            element_type = self.state.element_types[element.type]["Ename"]

            if element_type not in SURFACE_ELEMENT_TYPES:
                raise ValueError(f"Element type {element_type} is not a surface element type.")

            if type is None:
                type = element.type
            elif type != element.type:
                raise ValueError("Meshes containing multiple element types are not supported.")

            connectivity.append([node_number_to_index[node] for node in element.connectivity])
            elem_number_to_index[element_number] = element_index

        type = self.state.element_types[type]["Ename"]

        return (
            SurfaceMesh(
                nodes=nodes,
                type=type,
                connectivity=connectivity,
            ),
            node_number_to_index,
            elem_number_to_index,
        )

    def get_solid_model(self) -> tuple[SolidModel, dict[int, int], dict[int, int]]:
        """Translate the parsed solid APDL model to an OAMC solid model.

        Returns
        -------
        oamc.fem.SolidModel
            Translated OAMC solid model.
        dict of int to int
            Mapping from original node numbers to new node indices.
        dict of int to int
            Mapping from original element numbers to new element indices.

        Raises
        ------
        ValueError
            If
            - the model contains multiple element types,
            - the model contains multiple materials.
        """
        start = clock()

        nodes, node_number_to_index = self.get_nodes()

        type = None
        material_number = None
        connectivity = []
        elem_number_to_index: dict[int, int] = {}
        for index, (number, element) in enumerate(self.state.elements.items()):
            if self.state.element_types[element.type]["Ename"] not in SOLID_ELEMENT_TYPES:
                continue

            if type is None:
                type = element.type
            elif type != element.type:
                raise ValueError("Meshes containing multiple element types are not supported.")

            if material_number is None:
                material_number = element.material
            elif material_number != element.material:
                raise ValueError("Meshes containing multiple element materials are not supported.")

            connectivity.append([node_number_to_index[node] for node in element.connectivity])
            elem_number_to_index[number] = index

        type = self.state.element_types[type]["Ename"]
        material = self.state.materials[material_number]

        # Convert to Material instance:
        if "EY" in material:
            material = OrthotropicMaterial(
                E1=material["EX"],
                E2=material["EY"],
                E3=material["EZ"],
                nu12=material["PRXY"],
                nu23=material["PRYZ"],
                nu13=material["PRXZ"],
                G23=material["GYZ"],
                G13=material["GXZ"],
                G12=material["GXY"],
                rho=material["DENS"],
            )
        else:
            material = IsotropicMaterial(
                E=material["EX"],
                nu=material["NUXY"],
                rho=material["DENS"],
            )

        connectivity = numpy.array(
            connectivity,
            dtype=int,
        )

        dbc_renumbered = []
        for dbc in self.state.dbc:
            dbc_renumbered.append(
                BC(
                    node=node_number_to_index[dbc.node],
                    direction=dbc.direction,
                    value=dbc.value,
                )
            )

        nbc_renumbered = []
        for nbc in self.state.nbc:
            nbc_renumbered.append(
                BC(
                    node=node_number_to_index[nbc.node],
                    direction=nbc.direction,
                    value=nbc.value,
                )
            )

        logger.info(f"APDL model translated to OAMC model in {round(clock() - start, 3)} seconds.")

        return (
            SolidModel(
                mesh=SolidMesh(
                    nodes=nodes,
                    type=type,
                    connectivity=connectivity,
                ),
                material=material,
                dbc=dbc_renumbered,
                nbc=nbc_renumbered,
            ),
            node_number_to_index,
            elem_number_to_index,
        )


# APDL commands in alphabetic order (not considering leading slashes and
# stars):


@command("/AUX2")
def handle_slash_aux2(state: APDLState, args: list[str]) -> None:
    """/AUX2

    Enters the binary file dumping processor.
    """
    state.processor = APDLProcessor.AUX2


@command("/AUX3")
def handle_slash_aux3(state: APDLState, args: list[str]) -> None:
    """/AUX3

    Enters the results file editing processor.
    """
    state.processor = APDLProcessor.AUX3


@command("/AUX12")
def handle_slash_aux12(state: APDLState, args: list[str]) -> None:
    """
    /AUX12

    Enters the radiation processor.
    """
    state.processor = APDLProcessor.AUX12


@command("/AUX15")
def handle_slash_aux15(state: APDLState, args: list[str]) -> None:
    """
    /AUX15

    Enters the IGES file transfer processor.
    """
    state.processor = APDLProcessor.AUX15


@command("CMSEL")
def handle_cmsel(state: APDLState, args: list[str]) -> None:
    """
    CMSEL, Type, Name, Entity

    Selects a subset of components and assemblies.
    """

    selection_type = args[0].upper()

    if selection_type == "ALL":
        state.selected_components = set(state.components.keys())
        return

    if selection_type == "NONE":
        state.selected_components = set()
        return

    component_name = args[1].upper()

    if component_name:
        component = state.components[component_name]
        selection = component.entities

        try:
            target = {
                APDLEntityType.VOLUME: state.selected_volumes,
                APDLEntityType.AREA: state.selected_areas,
                APDLEntityType.LINE: state.selected_lines,
                APDLEntityType.KEYPOINT: state.selected_keypoints,
                APDLEntityType.ELEMENT: state.selected_elements,
                APDLEntityType.NODE: state.selected_nodes,
            }[component.type]
        except KeyError:
            raise ValueError(f"Unknown entity type in CMSEL: {component.type}")

    else:
        try:
            entity_type = {
                "VOLU": APDLEntityType.VOLUME,
                "AREA": APDLEntityType.AREA,
                "LINE": APDLEntityType.LINE,
                "KP": APDLEntityType.KEYPOINT,
                "ELEM": APDLEntityType.ELEMENT,
                "NODE": APDLEntityType.NODE,
            }[args[2]]
        except KeyError:
            raise ValueError(f"Unknown entity type in CMSEL: {args[2]}")

        selection = set()
        for name, component in state.components.items():
            if component.type == entity_type:
                selection.add(name)

        target = state.selected_components

    match selection_type:
        case "S":
            target.clear()
            target.update(selection)
        case "R":
            target.intersection_update(selection)
        case "A":
            target.update(selection)
        case "U":
            target.difference_update(selection)


@command("D")
def handle_d(state: APDLState, args: list[str]) -> None:
    """
    D, Node, Lab, VALUE, VALUE2, NEND, NINC, Lab2, Lab3, Lab4, Lab5, Lab6, MESHFLAG

    Defines degree-of-freedom constraints at nodes.
    """
    if len(args) > 3:
        raise NotImplementedError(
            "Arguments VALUE2, NEND, NINC, Lab2, Lab3, Lab4, Lab5, Lab6,"
            "and MESHFLAG of D command are not supported."
        )

    try:
        nodes = [int(args[0])]
    except ValueError:
        match nodes := args[0].upper():
            case "ALL":
                nodes = list(state.selected_nodes)
            case "P" | "PICK":
                raise NotImplementedError("Graphical picking is not supported.")
            case _:
                try:
                    nodes = state.components[nodes].entities
                except KeyError:
                    raise NotImplementedError(f"Component {nodes} does not exist.")

    label = args[1].upper()
    try:
        value = float(args[2])
    except Exception:
        value = 0

    match label:
        case "ALL":
            directions = (Direction.X, Direction.Y, Direction.Z)
        case "UX":
            directions = (Direction.X,)
        case "UY":
            directions = (Direction.Y,)
        case "UZ":
            directions = (Direction.Z,)
        case _:
            raise NotImplementedError(f"Degree of freedom {label} in D command is not supported.")

    for node in nodes:
        for direction in directions:
            state.dbc.append(
                BC(
                    node=node,
                    direction=direction,
                    value=value,
                )
            )


@command("E")
def handle_e(state: APDLState, args: list[str]) -> None:
    raise NotImplementedError("The E command is not yet implemented.")


@command("ESEL")
def handle_esel(state: APDLState, args: list[str]) -> None:
    """
    ESEL, Type, Item, Comp, VMIN, VMAX, VINC, KABS

    Selects a subset of elements.
    """

    # Determine selection type:
    if len(args) == 0 or args[0] == "":
        type = "ALL"
    else:
        type = args[0].upper()

    # Determine selection:
    if type in {"S", "R", "A", "U"}:
        if len(args) < 7:
            args += [""] * (7 - len(args))

        match item := args[1].upper():
            case "ELEM":
                if args[3]:
                    try:
                        selection = {int(args[3])}
                    except ValueError:
                        try:
                            selection = {state.variables[args[3]]}
                        except KeyError as error:
                            # TODO: Raise an Exception once expression evaluation is implemented.
                            logger.error(
                                f"Variable {args[3]} used in ESEL, ..., ELEM is not defined. Skipping it for now."
                            )
                            return
                else:
                    raise ValueError("Element number must be provided for ESEL, ..., ELEM.")
            case "TYPE":
                if args[3]:
                    etype = state.element_types[int(args[3])]["Ename"]
                    selection = {
                        element_number
                        for element_number, element in state.elements.items()
                        if element.type == etype
                    }
                else:
                    raise ValueError("Element type number must be provided for ESEL, ..., TYPE.")
            case "ENAME":
                if args[3]:
                    ename = {
                        154: ElementType.SURF154,
                        185: ElementType.SOLID185,
                        186: ElementType.SOLID186,
                        187: ElementType.SOLID187,
                        200: ElementType.MESH200,
                        285: ElementType.SOLID285,
                    }[int(args[3])]
                    selection = {
                        element_number
                        for element_number, element in state.elements.items()
                        if state.element_types[element.type]["Ename"] == ename
                    }
                else:
                    raise ValueError("Element type name must be provided for ESEL, ..., ENAME.")
            case _:
                raise NotImplementedError(f"ESEL item {item} is not yet implemented.")

    elif len(args) > 1:
        logger.warning(
            f"ESEL arguments Item, Comp, VMIN, VMAX, VINC, KABS are used only with argument Type = S, R, A, or U, not {type}."
        )

    # Select elements:
    match type:
        case "S":
            state.selected_elements = selection
        case "R":
            state.selected_elements &= selection
        case "A":
            state.selected_elements |= selection
        case "U":
            state.selected_elements -= selection
        case "ALL":
            state.selected_elements = set(state.elements.keys())
        case "NONE":
            state.selected_elements = set()
        case "INVE":
            state.selected_elements = set(state.elements.keys()) - state.selected_elements
        case "STAT":
            logger.INFO(f"Current element selection: {state.selected_elements}")
        case _:
            raise ValueError(f"Invalid ESEL type: {args[0]}")


@command("ESYS")
def handle_esys(state: APDLState, args: list[str]) -> None:
    """
    ESYS, KCN

    Sets the element coordinate system attribute pointer.
    """

    if len(args) < 1:
        kcn = 1
    else:
        kcn = int(args[0])

    state.attribute_pointers["ESYS"] = kcn


@command("ET")
def handle_et(state: APDLState, args: list[str]) -> None:
    """
    ET, ITYPE, Ename, KOP1, KOP2, KOP3, KOP4, KOP5, KOP6, INOPR
    """
    itype = int(args[0])
    element_type = {
        154: ElementType.SURF154,
        185: ElementType.SOLID185,
        186: ElementType.SOLID186,
        187: ElementType.SOLID187,
        200: ElementType.MESH200,
        285: ElementType.SOLID285,
    }.get(int(args[1]))
    state.element_types[itype] = {"Ename": element_type}
    for i, key_option in enumerate(args[2:8], start=1):
        if key_option:
            state.element_types[itype][f"KOP{i}"] = float(key_option)
    if len(args) > 8:
        state.element_types[itype]["INOPR"] = int(args[8])


@command("FINISH", "FINI")
def handle_finish(state: APDLState, args: list[str]) -> None:
    """
    FINISH

    Exits normally from a processor.
    """
    state.processor = APDLProcessor.BEGIN


@command("KEYOPT", "KEYOP", "KEYO")
def handle_keyopt(state: APDLState, args: list[str]) -> None:
    """
    KEYOPT, ITYPE, KNUM, VALUE

    Sets element key options.
    """
    try:
        itype = int(args[0])
    except ValueError as e:
        raise NotImplementedError(f"ITYPE = {args[0]} in command KEYOPT is not supported.") from e
    state.element_types[itype][f"KOP{args[1]}"] = float(args[2])


@command("MAT")
def handle_mat(state: APDLState, args: list[str]) -> None:
    """
    MAT, MAT

    Sets the element material attribute pointer.
    """

    if len(args) < 1:
        mat = 1
    else:
        mat = int(args[0])

    state.attribute_pointers["MAT"] = mat


@command("MP")
def handle_mp(state: APDLState, args: list[str]) -> None:
    """
    MP, Lab, MAT, C0, C1, C2, C3, C4
    """

    label = args[0].upper()
    if label in {"UVID", "UMID"}:
        return

    if len(args) > 3 and args[3] != "":
        raise NotImplementedError("Arguments C1, C2, C3, and C4 in command MP are not supported.")

    number = int(args[1])
    value = float(args[2])

    if number not in state.materials:
        state.materials[number] = {}

    if label not in ("UVID", "UMID"):
        state.materials[number][label] = value


@command("N")
def handle_n(state: APDLState, args: list[str]) -> None:
    raise NotImplementedError("The N command is not yet implemented.")


@command("NSEL")
def handle_nsel(state: APDLState, args: list[str]) -> None:
    """
    NSEL, Type, Item, Comp, VMIN, VMAX, VINC, KABS

    Selects a subset of nodes.
    """
    match args[0].upper():
        case "S":
            raise NotImplementedError("NSEL, S is not yet implemented.")
        case "R":
            raise NotImplementedError("NSEL, R is not yet implemented.")
        case "A":
            raise NotImplementedError("NSEL, A is not yet implemented.")
        case "U":
            raise NotImplementedError("NSEL, U is not yet implemented.")
        case "ALL":
            state.selected_nodes = set(state.nodes.keys())
        case "NONE":
            state.selected_nodes = set()
        case "INVE":
            state.selected_nodes = set(state.nodes.keys()) - state.selected_nodes
        case "STAT":
            logger.debug(f"NSEL, STAT: {state.selected_nodes}")
        case _:
            raise ValueError(f"Invalid NSEL type: {args[0]}")


@command("/PREP7")
def handle_slash_prep7(state: APDLState, args: list[str]) -> None:
    """
    /PREP7

    Enters the model creation preprocessor.
    """
    state.processor = APDLProcessor.PREP7


@command("/POST1")
def handle_slash_post1(state: APDLState, args: list[str]) -> None:
    """
    /POST1

    Enters the database results postprocessor.
    """
    state.processor = APDLProcessor.POST1


@command("/POST26")
def handle_slash_post26(state: APDLState, args: list[str]) -> None:
    """
    /POST26

    Enters the time-history results postprocessor.
    """
    state.processor = APDLProcessor.POST26


@command("REAL")
def handle_real(state: APDLState, args: list[str]) -> None:
    """
    REAL, NSET

    Sets the element real constant set attribute pointer.
    """

    if len(args) < 1:
        nset = 1
    else:
        nset = int(args[0])

    state.attribute_pointers["REAL"] = nset


@command("SECNUM")
def handle_secnum(state: APDLState, args: list[str]) -> None:
    """
    SECNUM, SECID

    Sets the element section attribute pointer.
    """

    if len(args) < 1:
        secid = 1
    else:
        secid = int(args[0])

    state.attribute_pointers["SECNUM"] = secid


@command("SFCONTROL", "SFCO")
def handle_sfcontrol(state: APDLState, args: list[str]) -> None:
    """
    SFCONTROL, KCSYS, LCOMP, VAL1, VAL2, VAL3, KTAPER, KUSE, KAREA, KPROJ, KFOLLOW

    Defines structural surface-load properties on selected elements and
    nodes for subsequent loading commands.
    """
    if args[0].upper() == "NONE":
        state.sfcontrol = {}
    else:
        try:
            state.sfcontrol = {
                "KCSYS": int(args[0]) if args[0] != "" else 0,
                "LCOMP": int(args[1]) if args[1] != "" else 0,
                "VAL1": int(args[2]) if args[2] != "" else 0,
                "VAL2": int(args[3]) if args[3] != "" else 0,
                "VAL3": int(args[4]) if args[4] != "" else 0,
                "KTAPER": int(args[5]) if len(args) > 5 and args[5] != "" else 0,
                "KUSE": int(args[6]) if len(args) > 6 and args[6] != "" else 0,
                "KAREA": int(args[7]) if len(args) > 7 and args[7] != "" else 0,
                "KRPOJ": int(args[8]) if len(args) > 8 and args[8] != "" else 0,
                "KFOLLOW": int(args[9]) if len(args) > 9 and args[9] != "" else 0,
            }
        except Exception as e:
            raise ValueError("Invalid arguments in SFCONTROL command.") from e


@command("SFE")
def handle_sfe(state: APDLState, args: list[str]) -> None:
    """
    SFE, Elem, LKEY, Lab, KVAL, VALUE1, VALUE2, VALUE3, VALUE4, MESHFLAG

    Defines surface loads on elements.
    """

    elem = args[0].upper()
    if elem in {"P", "PICK"}:
        raise NotImplementedError("Graphical picking is not supported.")
    if elem == "ALL":
        elements = list(state.selected_elements)
    else:
        try:
            elements = [int(elem)]
        except ValueError:
            try:
                elements = state.components[elem].entities
            except KeyError:
                raise ValueError(f"Component {args[0]} does not exist.")

    lkey = int(args[1])
    lab = args[2].upper()
    kval = int(args[3])

    match lab:
        case "PRES":
            match kval:
                case 0 | 1:
                    if len(args) > 5:
                        raise NotImplementedError(
                            f"Arguments VALUE2, VALUE3, VALUE4, MESHFLAG in command SFE, ..., PRES, {kval}, ... are not yet supported."
                        )
                    value1 = float(args[4])
                    for element in elements:
                        match etype := state.element_types[state.elements[element].type]["Ename"]:
                            case ElementType.SOLID186:
                                try:
                                    local_nodes = {
                                        1: [0, 3, 2, 1, 11, 10, 9, 8],
                                        2: [0, 1, 5, 4, 8, 17, 12, 16],
                                        3: [1, 2, 6, 5, 9, 18, 13, 17],
                                        4: [2, 3, 7, 6, 10, 19, 14, 18],
                                        5: [0, 4, 7, 3, 16, 15, 19, 11],
                                        6: [4, 5, 6, 7, 12, 13, 14, 15],
                                    }[lkey]
                                    nodes = state.elements[element].connectivity[local_nodes]
                                    weights = (
                                        -1 / 12,
                                        -1 / 12,
                                        -1 / 12,
                                        -1 / 12,
                                        1 / 3,
                                        1 / 3,
                                        1 / 3,
                                        1 / 3,
                                    )
                                    i = numpy.array(state.nodes[nodes[0]])
                                    j = numpy.array(state.nodes[nodes[1]])
                                    l = numpy.array(state.nodes[nodes[3]])
                                    area = numpy.linalg.norm(numpy.cross(j - i, l - i))
                                except KeyError:
                                    raise ValueError(
                                        f"Invalid face number {lkey} for SOLID186 element in SFE command."
                                    )
                            case _:
                                raise NotImplementedError(
                                    f"Pressure loads for {etype} elements are not yet supported."
                                )

                        for node, weight in zip(nodes, weights):
                            # TODO: Handle KCSYS etc. from SFCONTROL.
                            state.nbc.append(
                                BC(
                                    node=node,
                                    direction=Direction(state.sfcontrol["LCOMP"]),
                                    value=weight * value1 * area,
                                )
                            )
                case 2:
                    raise NotImplementedError(
                        "Imaginary pressure loads (SFE, ..., PRES, 2, ...) are not supported."
                    )
                case _:
                    raise ValueError(
                        f"Invalid value key (KVAL) {kval} in SFE command for PRES load."
                    )
        case _:
            raise NotImplementedError(
                f"Surface load label (Lab) {lab} in SFE command is not supported."
            )


@command("/SOLU")
def handle_slash_solu(state: APDLState, args: list[str]) -> None:
    """
    /SOLU

    Enters the solution processor.
    """
    state.processor = APDLProcessor.SOLUTION


@command("/TITLE")
def handle_slash_title(state: APDLState, args: list[str]) -> None:
    """
    /TITLE, Title
    """

    state.title = args[0]


@command("TYPE")
def handle_type(state: APDLState, args: list[str]) -> None:
    """
    TYPE, ITYPE

    Sets the element type attribute pointer.
    """

    if len(args) < 1:
        itype = 1
    else:
        itype = int(args[0])

    state.attribute_pointers["TYPE"] = itype


@command("/UNITS")
def handle_slash_units(state: APDLState, args: list[str]) -> None:
    """
    /UNITS, Label, LENFACT, MASSFACT, TIMEFACT, TEMPFACT, TOFFST, CHARGEFACT, FORCEFACT, HEATFACT
    """

    label = args[0]
    if label == "USER":
        raise NotImplementedError("User-defined unit systems are not supported.")
    state.units = label


# APDL block commands (see Mechanical APDL 2025 R2 Programmer's
# Reference Part 1, section 3.2) in alphabetic order (not considering
# leading slashes and stars):


@command("CMBLOCK", needs_iterator=True)
def handle_cmblock(
    state: APDLState,
    args: list[str],
    iterator: Iterator[tuple[int, str]],
) -> None:
    """
    CMBLOCK, Cname, Entity, NUMITEMS, , ,NUMBF, , KOPT \n
    Format

    CMBLOCK defines the entities contained in a node, element, or element face (EFACE) component.
    """

    name = args[0].upper()

    match args[1].upper():
        case "NODE":
            type = APDLEntityType.NODE
        case "ELEM" | "ELEMENT":
            type = APDLEntityType.ELEMENT
        case "EFACE":
            raise NotImplementedError("CMBLOCK with EFACE entity type is not supported.")
        case _:
            raise ValueError(f"Invalid CMBLOCK entity type: {args[1]}")

    entity_count = int(args[2])
    format = next(iterator)[1]
    entity_count_per_line = fields_per_line(format)
    line_count = (entity_count + entity_count_per_line - 1) // entity_count_per_line

    entities: list[int] = []
    for _ in range(line_count):
        line = next(iterator)[1]
        tokens = tokenize(line)
        for token in tokens:
            if (entity := int(token)) < 0:
                for i in range(1 + entities[-1], 1 - entity):
                    entities.append(i)
            else:
                entities.append(entity)

    state.components[name] = APDLComponent(
        type=type,
        entities=entities,
    )
    state.selected_components.add(name)


@command("EBLOCK", needs_iterator=True)
def handle_eblock(
    state: APDLState,
    args: list[str],
    iterator: Iterator[tuple[int, str]],
) -> None:
    """
    EBLOCK, NUM, KEY, ELMAX, ELSEL, CMPKEY, CMPNAME \n
    Format

    EBLOCK defines a block of elements.
    """

    num = int(args[0])

    if len(args) < 2:
        key = ""
    else:
        key = args[1]

    # Skip format line:
    next(iterator)

    # Read element data:
    match key.upper():
        case "SOLID":
            while not (line := next(iterator)[1]).strip().startswith("-1"):
                tokens = tokenize(line)
                number = int(tokens[10])
                if int(tokens[8]) > 8:
                    line = next(iterator)[1]
                    tokens.extend(tokenize(line))
                state.elements[number] = APDLElement(
                    material=int(tokens[0]),
                    type=int(tokens[1]),
                    real_constant_set=int(tokens[2]),
                    section=int(tokens[3]),
                    cosy=int(tokens[4]),
                    connectivity=numpy.array(
                        [int(node) for node in tokens[11:]],
                        dtype=int,
                    ),
                )
                state.selected_elements.add(number)
        case "COMPACT":
            node_count = NODE_COUNT_FROM_ELEMENT_TYPE[
                state.element_types[state.attribute_pointers["TYPE"]]["Ename"]
            ]
            read_next_line = node_count > num - 1
            while not (line := next(iterator)[1]).strip().startswith("-1"):
                tokens = tokenize(line)
                number = int(tokens[0])
                if read_next_line:
                    line = next(iterator)[1]
                    tokens.extend(tokenize(line))
                state.elements[number] = APDLElement(
                    material=state.attribute_pointers["MAT"],
                    type=state.attribute_pointers["TYPE"],
                    real_constant_set=state.attribute_pointers["REAL"],
                    section=state.attribute_pointers["SECNUM"],
                    cosy=state.attribute_pointers["ESYS"],
                    connectivity=numpy.array(
                        [int(node) for node in tokens[1:]],
                        dtype=int,
                    ),
                )
                state.selected_elements.add(number)
        case "":
            while not (line := next(iterator)[1]).strip().startswith("-1"):
                tokens = tokenize(line)
                number = int(tokens[0])
                # Field 2 is incorrectly labeled "type of section ID" in
                # the Mechanical APDL 2025 R2 Programmer's Reference,
                # section 3.2.9. Actually it is the element type number:
                type = int(tokens[1])
                real_constant_set = int(tokens[2])
                material = int(tokens[3])
                cosy = int(tokens[4])
                node_count = NODE_COUNT_FROM_ELEMENT_TYPE[state.element_types[type]["Ename"]]
                if node_count > num - 1:
                    line = next(iterator)[1]
                    tokens.extend(tokenize(line))
                state.elements[number] = APDLElement(
                    material=material,
                    type=type,
                    real_constant_set=real_constant_set,
                    section=state.attribute_pointers["SECNUM"],
                    cosy=cosy,
                    connectivity=numpy.array(
                        [int(node) for node in tokens[11:]],
                        dtype=int,
                    ),
                )
                state.selected_elements.add(number)
        case _:
            raise ValueError(f"Invalid EBLOCK format: {key}")


@command("NBLOCK", needs_iterator=True)
def handle_nblock(
    state: APDLState,
    args: list[str],
    iterator: Iterator[tuple[int, str]],
) -> None:
    """
    NBLOCK, NUMFIELD, Solkey, NDMAX, NDSEL, CMPKEY, CMPNAME, TSFOD, CSID, ORNTKEY \n
    Format

    NBLOCK defines a block of nodes.
    """

    if len(args) > 4:
        raise NotImplementedError(
            "Arguments CMPKEY, CMPNAME, TSFOD, CSID, ORNTKEY in NBLOCK are not supported."
        )

    if args[1] != "":
        raise NotImplementedError(f"Solkey {args[1]} in NBLOCK is not supported.")

    # Skip format line:
    next(iterator)

    # Read nodes:
    while not (line := next(iterator)[1]).strip().startswith("-1"):
        tokens = tokenize(line)
        number = int(tokens[0])
        state.nodes[number] = (
            float(tokens[1]),
            float(tokens[2]),
            float(tokens[3]),
        )
        state.selected_nodes.add(number)


# Ignored APDL commands in alphabetic order (not considering leading
# slashes and stars):

_IGNORED_COMMANDS = {
    "/BATCH",
    "CEWRITE",
    "CNTR",
    "/CONFIG",
    "DMPOPTION",
    "/EOF",
    "EQSL",
    "ETLIST",
    "/FCLEAN",
    "/FORMAT",
    "/GO",
    "/GOLIST",
    "/GOPR",
    "/GST",
    "/HEADER",
    "NLDIAG",
    "NLIST",
    "/NOLIST",
    "/NOPR",
    "/OUTPUT",
    "OUTRES",
    "/PAGE",
    "PRNSOL",
    "SHPP",
    "SOLVE",
    "/WB",
    "/XML",
    "XMLO",
}

for command_name in _IGNORED_COMMANDS:
    if command_name in _HANDLER_REGISTRY:
        raise Exception(
            f"APDL command {command_name} exists in the handler registry and in the set of ignored "
            "commands. One must be removed."
        )
