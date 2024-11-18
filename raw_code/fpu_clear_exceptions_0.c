void fpu_clear_exceptions(void)

{

    struct __attribute__((packed)) {

        uint16_t fpuc;

        uint16_t dummy1;

        uint16_t fpus;

        uint16_t dummy2;

        uint16_t fptag;

        uint16_t dummy3;

        uint32_t ignored[4];

        long double fpregs[8];

    } float_env32;



    asm volatile ("fnstenv %0\n" : : "m" (float_env32));

    float_env32.fpus &= ~0x7f;

    asm volatile ("fldenv %0\n" : : "m" (float_env32));

}
