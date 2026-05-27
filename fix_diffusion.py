with open("faster/networks/diffusion.py", "r") as f:
    content = f.read()

replacement = '''        if a.shape[0] != s_encoded.shape[0]:
            if a.shape[0] % s_encoded.shape[0] == 0:
                repeat_factor = a.shape[0] // s_encoded.shape[0]
                s_encoded = jnp.repeat(s_encoded, repeat_factor, axis=0)

        if a.shape[0] != cond.shape[0]:
            if a.shape[0] % cond.shape[0] == 0:
                repeat_factor = a.shape[0] // cond.shape[0]
                cond = jnp.repeat(cond, repeat_factor, axis=0)
                
        reverse_input = jnp.concatenate([a, s_encoded, cond], axis=-1)'''

content = content.replace('        reverse_input = jnp.concatenate([a, s_encoded, cond], axis=-1)', replacement)

with open("faster/networks/diffusion.py", "w") as f:
    f.write(content)
