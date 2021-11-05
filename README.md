# STL10-ResNET101

## Intuition behind Residual blocks:
### If the identity mapping is optimal, We can easily push the residuals to zero (F(x) = 0) than to fit an identity mapping (x, input=output) by a stack of non-linear layers. In simple language it is very easy to come up with a solution like F(x) =0 rather than F(x)=x using stack of non-linear cnn layers as function (Think about it). So, this function F(x) is what the authors called Residual function.


![This is an image](https://miro.medium.com/max/856/1*WVs9ywVLLKjSUBZ_mnfFrw.png)
