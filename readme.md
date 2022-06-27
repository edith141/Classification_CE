  
# ML
An implementation of Linear Classification (Logistic Regression) from scratch.

# Logistic Regression implementation
This is (a supervised learning algo) used to classify (or "predict classification") items/data to a class from the given possible classes. Here, we are using binary classification, i.e. we'll predict if a data/item belongs to a given class or not. One simple way to see this is as an answer to the question "Does this data item belong to this class?" The answer could either be True or False (binary).

Now, as we did in LR, we still try and fit a line to the given data points, but in case of classification, the data is not linearly separable, as shown in this image. 

The solution is rather simple, we use a **logistic** (sigmoid) function. This also uses an equation as it's representation. The input value is combined with the coefficients (wts and bias) to predict an o/p- y.

The key difference here to note is the o/p is either 0 or 1 (binary).

The sigmoid function is of the form: 
$$y =g(z) = \frac{1}{1 + e^-z} $$

![sigmoid_func](/sigmoid_func.png)

Now, in terms of input "x" and coeff "w", we can write this as:
$$h_w(x) =g(w.x) = \frac{1}{1 +e^(-w.x^)} $$
When the data item is near 0.5, it is almost in the middle of the boundary of the two classes. Similarly, if it moves away from the center, it approaces 0 or 1. This may be seen as the probability of that data point belonging to that class.
*Here too, w is the weight (coefficient of x) and b is the bias.*

Fitting of these wts for the model to produce as accurate results (low error) as possible is logistic regression. Now, we again use GD to minimise the cost function in this.
## Gradient descent
The GD works just as before, the change being in the cost funtion.

(Note: since it is a probability of a point belonging to a class, we can do loss = $1 - h_w(x)$ too.)

So, for GD, we get the cost function as:
$$\frac{\delta}{\delta w_i} Loss(w)=\frac{\delta}{\delta w_i}(y - h_w(x))^2$$
Applying chain rule $$\frac{\delta_g(f(x))}{\delta x} = g' (f(x))\frac{\delta f(x)}{\delta x}$$

... solving this we get-  $$-2(y-h_w(x)) * g'(w.x)*x_i$$
(where, g'(f(x)) is the derivative of the outer function)

And, derivative of a logistic fn- g(z) satisfies $g'(z) = g(z) . (1-g(z))$  

now, plugging in the values of the logistic fn (in h(x)) we get the equation to update the weight for minimizing the loss - 
$$w_i \leftarrow w_i + lr(y-h_w(x)) * (h_w(x)).(1-(y-h_w(x)) * x_i$$
The same equation is used in the code as well to update the coeff.


<hr>

### The data:
Provided in the file. Earthquake magnitude data and underground explosions data; predict (classify) if an event was an earthquake or not (0 or 1).
The data set, in each run, is first randomized. Then the data set is split in subsets - train dataset, and the test dataset. This is repeated 5 times for 5 individual runs of the model to get a better idea of the performace of the model.

The algorithm will try and find (and optimise) the weight and bias. At the end, we should have an equation that outputs 0/1.


## The model and results
The algorithm/model improves the pridiction with each iteration (optimises wt and bias) moving towards the direction suggested by the slope of the gradient. The error rate is shown here.
ERROR RATE IMAGE!
![initial state](/error_rate_ss.png)


## Calculating the avg accuracy of the model with 5 different (random) subsets of train and test data.
The avg accuracy I get with different runs is almost always above 95%. Sometimes however, it goes near 90% and even fewer times, near 85%. The low accuracy is best explained by the random subsets of possibly skewed datasets being created and thus affecting the accuracy of the model.

![initial state](/res1.png)
![initial state](/res2.png)
![initial state](/res3.png)
