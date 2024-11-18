static void cpu_print_cc(FILE *f,

                         int (*cpu_fprintf)(FILE *f, const char *fmt, ...),

                         uint32_t cc)

{

    cpu_fprintf(f, "%c%c%c%c", cc & PSR_NEG? 'N' : '-',

                cc & PSR_ZERO? 'Z' : '-', cc & PSR_OVF? 'V' : '-',

                cc & PSR_CARRY? 'C' : '-');

}
