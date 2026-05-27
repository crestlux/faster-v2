import re
with open("faster/agents/expo_learner.py", "r") as f:
    content = f.read()

content = re.sub(r'r_observations = jnp.repeat\(jnp.expand_dims\(observations\[0\], axis=0\), self.ne_samples, axis=0\)', 'r_observations = jnp.expand_dims(observations[0], axis=0)', content)
content = re.sub(r'observations_repeated = jnp.repeat\(observations, self.train_N, axis=0\)', 'observations_repeated = observations', content)
content = re.sub(r'observations_repeated = jnp.repeat\(observations, self.train_N \+ self.ne_samples_train, axis=0\)', 'observations_repeated = observations', content)
content = re.sub(r'r_observations = jnp.repeat\(observations, self.ne_samples_train, axis=0\).*?\n', 'r_observations = observations\n', content)
content = re.sub(r'observations = jax.tree_map.*?else jnp.repeat\(x, self.N, axis=0\), observations\)', 'pass # observations = observations', content)


with open("faster/agents/expo_learner.py", "w") as f:
    f.write(content)
