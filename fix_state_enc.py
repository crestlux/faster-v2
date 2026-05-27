with open("faster/networks/state_action_value.py", "r") as f:
    content = f.read()

enc_replace = '''if actions.shape[0] != obs_encoded.shape[0]:
            if actions.shape[0] % obs_encoded.shape[0] == 0:
                repeat_factor = actions.shape[0] // obs_encoded.shape[0]
                obs_encoded = jnp.repeat(obs_encoded, repeat_factor, axis=0)
        inputs = jnp.concatenate([obs_encoded, actions], axis=-1)'''

# We need to replace the second match or replace both
# It's at the end of StateActionEncoder
content = content.replace('        inputs = jnp.concatenate([obs_encoded, actions], axis=-1)', enc_replace)
with open("faster/networks/state_action_value.py", "w") as f:
    f.write(content)
