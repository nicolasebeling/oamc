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

Standard material definition:

.. code-block:: apdl

    /wb,mat,start              !  starting to send materials
    /com,*********** Send Materials ***********
    Temperature = 'TEMP' ! Temperature
    MP,DENS,1,1.14e-09,	! tonne mm^-3
    MP,EX,1,1111,	! tonne s^-2 mm^-1
    MP,NUXY,1,0.3499,
    MP,ALPX,1,0.0001467,	! C^-1
    MP,KXX,1,0.2428,	! tonne mm s^-3 C^-1
    MP,C,1,1500000000,	! mm^2 s^-2 C^-1
    MP,RSVX,1,1.834e+15,	! ohm mm
    MP,LSST,1,0.1171,
    MP,PERX,1,14.38,
    MP,UVID,1,069da812-d02a-4266-a6ab-3b9b393f0615
    MP,UMID,1,e810dba8-e0a5-4943-89d8-e1fc59632241

    /wb,mat,end                !  done sending materials
