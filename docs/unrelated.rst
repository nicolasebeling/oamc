Unrelated
=========

Snippets and notes that were useful during development but are not directly related to the OAMC package anymore.

APDL Snippets
-------------

Export nodes, element types, element connectivity, and stresses from Ansys Mechanical:

.. code-block:: apdl

    /POST1
    SET, 1

    ! Select supported elements:
    ALLSEL
    ESEL, S, ENAME, , 185 ! SOLID185 (linear hex element)
    ESEL, A, ENAME, , 186 ! SOLID186 (quadratic hex element)
    ESEL, A, ENAME, , 187 ! SOLID187 (quadratic tet element)
    ESEL, A, ENAME, , 285 ! SOLID186 (linear tet element)

    ! Select all nodes of the currently selected elements:
    NSLE

    ! Refine the selection to contain only corner nodes:
    NSLE,R,CORNER

    ! Prevent page breaks:
    /PAGE, , , 1E9, 240,

    ! 15 characters per column, 6 of which are decimal digits:
    /FORMAT, , , 15, 6, ,

    ! Disable summaries:
    /HEADER, off, off, off, off, on, off

    ! Output node coordinates:
    /OUTPUT, nodes, txt
    NLIST, , , , COORD
    /OUTPUT

    ! Output element types:
    /OUTPUT, types, txt
    ETLIST
    /OUTPUT

    ! Output element connectivity:
    /OUTPUT, elements, txt
    ELIST, ALL, , , 0, 0
    /OUTPUT

    ! Output stresses:
    /OUTPUT, stresses, txt
    PRNSOL, S
    /OUTPUT
