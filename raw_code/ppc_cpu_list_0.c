void ppc_cpu_list (FILE *f, int (*cpu_fprintf)(FILE *f, const char *fmt, ...))

{

    int i, max;



    max = ARRAY_SIZE(ppc_defs);

    for (i = 0; i < max; i++) {

        (*cpu_fprintf)(f, "PowerPC %-16s PVR %08x\n",

                       ppc_defs[i].name, ppc_defs[i].pvr);

    }

}
