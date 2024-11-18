static void nvdimm_build_common_dsm(Aml *dev)

{

    Aml *method, *ifctx, *function, *dsm_mem, *unpatched, *result_size;

    uint8_t byte_list[1];



    method = aml_method(NVDIMM_COMMON_DSM, 4, AML_SERIALIZED);

    function = aml_arg(2);

    dsm_mem = aml_name(NVDIMM_ACPI_MEM_ADDR);



    /*

     * do not support any method if DSM memory address has not been

     * patched.

     */

    unpatched = aml_if(aml_equal(dsm_mem, aml_int(0x0)));



    /*

     * function 0 is called to inquire what functions are supported by

     * OSPM

     */

    ifctx = aml_if(aml_equal(function, aml_int(0)));

    byte_list[0] = 0 /* No function Supported */;

    aml_append(ifctx, aml_return(aml_buffer(1, byte_list)));

    aml_append(unpatched, ifctx);



    /* No function is supported yet. */

    byte_list[0] = 1 /* Not Supported */;

    aml_append(unpatched, aml_return(aml_buffer(1, byte_list)));

    aml_append(method, unpatched);



    /*

     * The HDLE indicates the DSM function is issued from which device,

     * it is not used at this time as no function is supported yet.

     * Currently we make it always be 0 for all the devices and will set

     * the appropriate value once real function is implemented.

     */

    aml_append(method, aml_store(aml_int(0x0), aml_name("HDLE")));

    aml_append(method, aml_store(aml_arg(1), aml_name("REVS")));

    aml_append(method, aml_store(aml_arg(2), aml_name("FUNC")));



    /*

     * tell QEMU about the real address of DSM memory, then QEMU

     * gets the control and fills the result in DSM memory.

     */

    aml_append(method, aml_store(dsm_mem, aml_name("NTFI")));



    result_size = aml_local(1);

    aml_append(method, aml_store(aml_name("RLEN"), result_size));

    aml_append(method, aml_store(aml_shiftleft(result_size, aml_int(3)),

                                 result_size));

    aml_append(method, aml_create_field(aml_name("ODAT"), aml_int(0),

                                        result_size, "OBUF"));

    aml_append(method, aml_concatenate(aml_buffer(0, NULL), aml_name("OBUF"),

                                       aml_arg(6)));

    aml_append(method, aml_return(aml_arg(6)));

    aml_append(dev, method);

}
