if True:
    import itertools
    timesteps = [1,2,3]
    residual = [True, False]
    leaky_rely = [True, False]
    unique_attention_projection = [True, False]
    opattention = [True, False]
    spaces = ['PNAS', 'nb201', 'Amoeba']
    configurations = list(itertools.product(timesteps, residual, leaky_rely, unique_attention_projection, opattention, spaces))
    base_command = (
        'False,arch_abl,True,30000,4,python new_main.py --seed 42 '
        '--name_desc arch_abl --gnn_type ensemble --sample_sizes 8 16 32 --batch_size 8  '
        '--representation adj_gin  --space {space} --timesteps {timesteps} '
    )
    for config in configurations:
        timesteps, residual, leaky_rely, unique_attention_projection, opattention, space = config
        command = base_command.format(space=space, timesteps=timesteps)
        if not residual: 
            command += ' --no_residual'
        if not leaky_rely: 
            command += ' --no_leaky_rely'
        if not unique_attention_projection: 
            command += ' --no_unique_attention_projection'
        if not opattention: 
            command += ' --no_opattention'
        print(command)