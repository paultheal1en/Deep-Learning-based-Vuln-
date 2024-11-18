void test_fenv(void)

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

    struct __attribute__((packed)) {

        uint16_t fpuc;

        uint16_t fpus;

        uint16_t fptag;

        uint16_t ignored[4];

        long double fpregs[8];

    } float_env16;

    double dtab[8];

    double rtab[8];

    int i;



    for(i=0;i<8;i++)

        dtab[i] = i + 1;



    TEST_ENV(&float_env16, "data16 fnstenv", "data16 fldenv");

    TEST_ENV(&float_env16, "data16 fnsave", "data16 frstor");

    TEST_ENV(&float_env32, "fnstenv", "fldenv");

    TEST_ENV(&float_env32, "fnsave", "frstor");



    /* test for ffree */

    for(i=0;i<5;i++)

        asm volatile ("fldl %0" : : "m" (dtab[i]));

    asm volatile("ffree %st(2)");

    asm volatile ("fnstenv %0\n" : : "m" (float_env32));

    asm volatile ("fninit");

    printf("fptag=%04x\n", float_env32.fptag);

}
