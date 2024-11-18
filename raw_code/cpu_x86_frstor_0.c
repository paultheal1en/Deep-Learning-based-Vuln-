void cpu_x86_frstor(CPUX86State *s, uint8_t *ptr, int data32)

{

    CPUX86State *saved_env;



    saved_env = env;

    env = s;

    

    helper_frstor(ptr, data32);



    env = saved_env;

}
