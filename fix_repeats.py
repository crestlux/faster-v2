import re

def process_file(file_path):
    with open(file_path, "r") as f:
        content = f.read()

    # faster_expo_learner
    if "faster_expo_learner.py" in file_path:
        # replace observations_repeated = jax.tree_map(lambda x: jnp.repeat(x, agent.N, axis=0), observations)
        content = re.sub(r'observations_repeated = jax.tree_map\(lambda x: jnp.repeat[^)]+\), observations\)', 'observations_repeated = observations', content)
        content = re.sub(r'obs_rep = jax.tree_map\(lambda x: jnp.repeat[^)]+\), observations\)', 'obs_rep = observations', content)
        content = re.sub(r'r_obs = jax.tree_map\(lambda x: jnp.repeat[^)]+\), observations\)', 'r_obs = observations', content)
        content = re.sub(r'r_observations = jax.tree_map\(lambda x: jnp.repeat[^)]+\), observations\)', 'r_observations = observations', content)
        
    elif "faster_idql_learner.py" in file_path:
        content = re.sub(r'observations_repeated = jax.tree_map\(lambda x: jnp.repeat[^)]+\), observations\)', 'observations_repeated = observations', content)
        content = re.sub(r'obs_rep = jax.tree_map\(lambda x: jnp.repeat[^)]+\), observations\)', 'obs_rep = observations', content)

    with open(file_path, "w") as f:
        f.write(content)

process_file("faster/agents/faster_expo_learner.py")
process_file("faster/agents/faster_idql_learner.py")
process_file("faster/agents/expo_learner.py")
process_file("faster/agents/idql_learner.py")
