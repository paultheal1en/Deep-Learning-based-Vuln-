void helper_stsw(CPUPPCState *env, target_ulong addr, uint32_t nb,

                 uint32_t reg)

{

    int sh;



    for (; nb > 3; nb -= 4) {

        cpu_stl_data(env, addr, env->gpr[reg]);

        reg = (reg + 1) % 32;

        addr = addr_add(env, addr, 4);

    }

    if (unlikely(nb > 0)) {

        for (sh = 24; nb > 0; nb--, sh -= 8) {

            cpu_stb_data(env, addr, (env->gpr[reg] >> sh) & 0xFF);

            addr = addr_add(env, addr, 1);

        }

    }

}
