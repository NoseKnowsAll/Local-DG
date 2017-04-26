How to visualize data in Paraview:


You must have Paraview 4.4.0 installed on your actual computer. For step-by-step instructions, go to 
http://www.nersc.gov/users/data-analytics/data-visualization/paraview-2/
[Note: You do not have to connect paraview to Edison for this README. But if you do not, then you have to SCP the data to your physical computer every time you create data before you can visualize]


In the code, ensure that you are calling initXYZVFile("output/filename.txt", "OutputNameToView"); at the beginning to setup the output file.
Then you can call exportToXYZVFile("output/filename.txt", mesh.globalCoords, propertyToOutput); when you want to append some data to that file.


Now open Paraview (and either connect to Edison or SCP data to your local machine).
File->Open->output/filename.txt
Under properties, ensure "Use String Delimiter", "Have Headers", "Merge Consecutive Delimiters" are all checked
Field Delimiter Characters = ","
Select Apply (Paraview should immediately open a spreadsheet with the values that can be safely ignored/closed)

With the object in Pipeline Browser selected, Filters->Alphabetical->Table To Points
Choose appropriate X/Y/Z columns
Select Apply (Paraview should immediately open a render view with the point cloud)

*With the point cloud object in Pipeline Browser selected, Filters->Alphabetical->Delaunay 3D
Select Apply (Paraview should compute and then view a volume surrounding the point cloud)
[Note: The DG mesh is not actually a good mesh as-is because of the overlapping points. It's OK so far, but maybe this will be a big factor later?]

With the Delaunay object in Pipeline Browser selected, Filters->Alphabetical->Countour
Choose value range to be the isosurface(s) you want to plot [Default is 1 iso at mean(data)]
Select Apply (Paraview should immediately view the isosurface)


Optional step (adding color)
With the contour object in Pipeline Browser selected, change "Coloring" to "Normals"

*Optional step (Glyphs instead of Delaunay)
With the point cloud object in Pipeline Browser selected, Filters->Alphabetical->Glyph
"Glyph Type" = "Sphere" makes the most sense for scalar data, "Arrow" for vector data