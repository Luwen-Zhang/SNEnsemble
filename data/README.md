# Copyright Notification

## Composites

We reformat and merge fatigue testing records from SNL/MSU/DOE, OptiMat, UPWIND, and FACT databases into a single database with careful considerations, including filtering invalid entries, rename variables, fix improper calculations in the records, to obtain clean and consistent data to fit machine 
learning usages. We leave the original format ("2019_SNL_MSU_DOE_Composite-Materials-Database_Wind_29pt0.xlsx" and "Optidat UPWIND 28_3_2017.xls") in the folder. Any derived database should be directly based on the original files instead of our merged database. We are not responsible for the accuracy of the testing records.

We acknowledge Montana State University, Knowledge Centre Wind turbine Materials and Construction, and Mr. Sibrand Raijmaekers (LM Wind Power) for the databases.

Following are copyright contents from the original databases.

### SNL/MSU/DOE database

```
SNL/MSU/DOE COMPOSITE MATERIAL FATIGUE DATABASE
Mechanical Properties of Composite Materials for Wind Turbine Blades
Version 29.0; May 13, 2019
Montana State University - Bozeman

This database was prepared as a part of  work sponsored by an agency of the U.S. Government.
Neither the U.S. Government, nor any agency thereof, nor any of  their employees, nor any of their contractors,
subcontractors, or their employees, makes any warranty, expressed or implied, or assumes any legal liability or
responsibility for the accuracy, completeness, or usefulness of this program, or represents that opinion expressed
herein do not necessarily state or reflect those of the U.S. Government, any agency thereof or any of their 
contractors or subcontractors. The material presented in this publication should not be used or relied upon for 
any specific application without competent examination and verification of its accuracy, suitability, and 
applicability by qualified professionals. Reference herein to any specific commercial product or process 
by trade name, trademark, manufacturer, or otherwise, does not necessarily constitute or imply its endorsement 
or recommendation. This 29.0 version of the database supersedes all previous versions.
This database is maintained by Daniel Samborsky (DanielS@montana.edu, 406-994-7186) at Montana State University
Updates, reports and conference papers can be downloaded from www.montana.edu/composites or windpower.sandia.gov/materials-reliability
```

### Optidat UPWIND database

```
Copyright notice:

This database is Copyright (C) 2007 by the Knowledge Centre Wind turbine Materials and Constructions (KC-WMC).

Permission is granted to use this work in the following ways:

      1) You may make private copies for your own personal use.
      2) You may create derivative works for your own personal use.
      3) You may publish short excerpts from the database, provided that a reference is included to: "Nijssen, R.P.L., ‘OptiDAT – fatigue of wind turbine materials database’, regular updates via www.kc-wmc.nl"
      4) You may distribute this work or make it available for copying by
         others only if ALL of the following three conditions are met:
         a) The information content of the database is unchanged.
            (You may not add, delete, or modify records.  You may, however, reformat and/or reorder the data.)
         b) The distribution is not made for monetary or material gain.
         c) A copy of this notice is included with every distributed copy of the database.

Disclaimer:  The database is supplied without any warranty, stated or implied. In particular, no claim is made that it is accurate, complete, or suitable for any purpose.  Use it at your own risk.
```

## Alloys

The datasets of additively manufactured alloys and complex metallic alloys are extracted from papers of Zhang et al. These data points are extracted from numerous research papers collected by the authors. For the complex metallic alloys dataset, we also extract fractions of chemical elements as individual features. Any derived database should be directly based on the original files instead of our modified ones. We are not responsible for the accuracy of the testing records.

* Zhang, Z., Tang, H. & Xu, Z. Fatigue database of complex metallic alloys. *Sci Data* **10**, 447 (2023). https://doi.org/10.1038/s41597-023-02354-1
* Zhang, Z., Xu, Z. Fatigue database of additively manufactured alloys. *Sci Data* **10**, 249 (2023). https://doi.org/10.1038/s41597-023-02150-x

We select parts of these data:

* Not runout;
* Stress-Life data (S-N);
* Uniaxial loading;
* Load ratio (R-value) is not empty.

Note that if two values are recorded for a feature, the final value is the average of these two values.

These datasets follow [CC BY 4.0 License](https://creativecommons.org/licenses/by/4.0/).
