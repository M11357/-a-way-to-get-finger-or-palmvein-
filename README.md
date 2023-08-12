#
一种基于主曲率理论的静脉提取方法



A Method for Extracting Veins Based on Principal Curvature Theory



【531】,{531clahe},(531ssr)分别对应【原图】，{使用clahe增强后的原图}，以及使用（ssr方法增强过的原图）



【531】，{531 clahe} ，（531 ssr） respectively correspond to the 【original image】, the {original image enhanced with clahe}, and the （original image enhanced with ssr method）



main2.py是采用最小二乘法做的曲线拟合，进而找到曲率最大点，main.py则采用了b样条曲线



main2. py is a curve fitting using the least squares method to find the point of maximum curvature, while main.py uses a b-spline curve



main2长于稳健性，在【531】,{531clahe},(531ssr)上的表现都差不多，而且不会像main一样把大量毛细血管导出来；main的好处是，在常规边缘检测项目中有更好的表现。



main2 excels in robustness and performs similarly on [531], {531clahe}, and (531ssr), and does not lead out a large number of capillaries like main; The advantage of main is that it performs better in conventional edge detection projects.



darktest是某调暗了的手背静脉图像，同时附两张效果图。



darktest is an image of a darkened vein on the back of a hand, accompanied by two renderings.



非常明显地，使用main2这种方法，可以省去图像增强的步骤，理论上做好图像降噪就可以艹图了。



It is very obvious that using the main2 method can eliminate the step of image enhancement, and theoretically, image denoising can be done well.

