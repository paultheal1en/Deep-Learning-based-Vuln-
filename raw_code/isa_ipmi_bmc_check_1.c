static void isa_ipmi_bmc_check(Object *obj, const char *name,

                               Object *val, Error **errp)

{

    IPMIBmc *bmc = IPMI_BMC(val);



    if (bmc->intf)

        error_setg(errp, "BMC object is already in use");

}
