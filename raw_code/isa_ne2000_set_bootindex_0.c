static void isa_ne2000_set_bootindex(Object *obj, Visitor *v,

                                     const char *name, void *opaque,

                                     Error **errp)

{

    ISANE2000State *isa = ISA_NE2000(obj);

    NE2000State *s = &isa->ne2000;

    int32_t boot_index;

    Error *local_err = NULL;



    visit_type_int32(v, name, &boot_index, &local_err);

    if (local_err) {

        goto out;

    }

    /* check whether bootindex is present in fw_boot_order list  */

    check_boot_index(boot_index, &local_err);

    if (local_err) {

        goto out;

    }

    /* change bootindex to a new one */

    s->c.bootindex = boot_index;



out:

    if (local_err) {

        error_propagate(errp, local_err);

    }

}
