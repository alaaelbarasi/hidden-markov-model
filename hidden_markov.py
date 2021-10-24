import tensorflow as tf
import tensorflow_probability as tfp
tfd=tfp.distributions
initial_distribution = tfd.Categorical(probs=[0.2, 0.8])  # Refer to point 2 above
transition_distribution = tfd.Categorical(probs=[[0.5, 0.5],
                                                 [0.2, 0.8]])  # refer to points 3 and 4 above
observation_distribution = tfd.Normal(loc=[0., 15.], scale=[5., 10.])
#creating the model 
model= tfd.HiddenMarkovModel(
    initial_distribution=initial_distribution,
     transition_distribution=transition_distribution, 
     observation_distribution=observation_distribution,
    num_steps=7
)
mean=model.mean()
with tf.compat.v1.Session() as sess:  
  print(mean.numpy())