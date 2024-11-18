__org_qemu_x_Union1 *qmp___org_qemu_x_command(__org_qemu_x_EnumList *a,

                                              __org_qemu_x_StructList *b,

                                              __org_qemu_x_Union2 *c,

                                              __org_qemu_x_Alt *d,

                                              Error **errp)

{

    __org_qemu_x_Union1 *ret = g_new0(__org_qemu_x_Union1, 1);



    ret->type = ORG_QEMU_X_UNION1_KIND___ORG_QEMU_X_BRANCH;

    ret->u.__org_qemu_x_branch = strdup("blah1");







    return ret;
