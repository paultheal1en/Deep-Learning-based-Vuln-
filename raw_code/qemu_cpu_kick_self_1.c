void qemu_cpu_kick_self(void)

{

#ifndef _WIN32

    assert(cpu_single_env);



    raise(SIG_IPI);

#else

    abort();

#endif

}
