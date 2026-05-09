User Guide
==========

The following workflow has been tested in Ansys Workbench 2025 R2. Steps may change in future releases.

Open Ansys Workbench.

In Ansys Workbench
------------------

Drag a **Static Structural** analysis system into the project schematic.

Right-click on **Geometry**, select **Import Geometry**, select **Browse...**, then select your STEP file.

Right-click on **Model**, select **Edit...** and wait for Ansys Mechanical to start.

In Ansys Mechanical
-------------------

Right-click on **Static Structural** in the Outline, select **Insert**, select **Displacement**, then apply your displacement boundary condition. Repeat if you need multiple.

Right-click on **Static Structural** in the Outline, select **Insert**, select **Force**, then apply your pressure load. Change **Applied By** from **Surface Effect** to **Direct**. This simplifies parsing the ds.dat file. Repeat if you need multiple.

.. note::

    **Fixed Support**, **Displacement** and **Force** on faces of solid elements are the only boundary conditions that are currently supported by OAMC. **Bearing Load** (a directed pressure load) is available too, but only by creating a named selection and applying the load manually in Python. See examples 1 and 3 for details.

    If additional boundary conditions such as remote forces and displacements are needed, functions for parsing them from ds.dat must be added in :file:`src/oamc/integrations/ansys/parser.py`.

Enter the **Element Face** selection mode in the Graphics Toolbar, select one element face of the face that will be placed on the mold (or multiple on different parts of the face if it is strongly curved or has kinks), select **Extend** in the Graphics Toolbar, select **Limits**, right-click on selected region, select **Create Named Selection**, enter the name "Mold", hit **OK**. Select the named selection in the Outline and change **Send As** from **Nodes** to **Mesh 200**.

You may want to insert a few displacement, strain, or stress results under **Solution** to check if the boundary conditions you applied lead to physically plausible results.

Hit **Solve** at least once to create the solver input (ds.dat) file that we need for OAMC and run the linear static analysis.

Check the results you inserted to ensure the results roughly match your intuition or, ideally, analytical estimates.

Save the project and exit Ansys Mechanical and Ansys Workbench.

In Python
---------

Set up a Python project as explained on the :doc:`./install` page. Create a new script. You can use :file:`examples/template` as a starting point. Navigate to the directory where you saved the Workbench file and copy :file:`<file name>_files\\dp0\\SYS\\MECH\\ds.dat` to the directory that contains your Python script. Then, try to run the script.

If you have questions, feel free to ask!
