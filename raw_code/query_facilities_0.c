static void query_facilities(void)

{

    unsigned long hwcap = qemu_getauxval(AT_HWCAP);



    /* Is STORE FACILITY LIST EXTENDED available?  Honestly, I believe this

       is present on all 64-bit systems, but let's check for it anyway.  */

    if (hwcap & HWCAP_S390_STFLE) {

        register int r0 __asm__("0");

        register void *r1 __asm__("1");



        /* stfle 0(%r1) */

        r1 = &facilities;

        asm volatile(".word 0xb2b0,0x1000"

                     : "=r"(r0) : "0"(0), "r"(r1) : "memory", "cc");

    }

}
