#
一种基于主曲率理论的静脉提取方法



A Method for Extracting Veins Based on Principal Curvature Theory



【531】,{531clahe},(531ssr)分别对应【原图】，{使用clahe增强后的原图}，以及使用（ssr方法增强过的原图）



【531】，{531 clahe} ，（531 ssr） respectively correspond to the 【original image】, the {original image enhanced with clahe}, and the （original image enhanced with ssr method）



main2.py是采用最小二乘法做的曲线拟合，进而找到曲率最大点，main.py则采用了b样条曲线



main2. py is a curve fitting using the least squares method to find the point of maximum curvature, while main.py uses a b-spline curve









