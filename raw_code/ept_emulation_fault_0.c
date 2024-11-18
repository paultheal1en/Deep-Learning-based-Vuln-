static bool ept_emulation_fault(uint64_t ept_qual)

{

    int read, write;



    /* EPT fault on an instruction fetch doesn't make sense here */

    if (ept_qual & EPT_VIOLATION_INST_FETCH) {

        return false;

    }



    /* EPT fault must be a read fault or a write fault */

    read = ept_qual & EPT_VIOLATION_DATA_READ ? 1 : 0;

    write = ept_qual & EPT_VIOLATION_DATA_WRITE ? 1 : 0;

    if ((read | write) == 0) {

        return false;

    }



    /*

     * The EPT violation must have been caused by accessing a

     * guest-physical address that is a translation of a guest-linear

     * address.

     */

    if ((ept_qual & EPT_VIOLATION_GLA_VALID) == 0 ||

        (ept_qual & EPT_VIOLATION_XLAT_VALID) == 0) {

        return false;

    }



    return true;

}
