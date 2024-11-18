static void cpudef_init(void)

{

#if defined(cpudef_setup)

    cpudef_setup(); /* parse cpu definitions in target config file */

#endif

}
