type_init(vmgenid_register_types)



GuidInfo *qmp_query_vm_generation_id(Error **errp)

{

    GuidInfo *info;

    VmGenIdState *vms;

    Object *obj = find_vmgenid_dev();



    if (!obj) {


        return NULL;

    }

    vms = VMGENID(obj);



    info = g_malloc0(sizeof(*info));

    info->guid = qemu_uuid_unparse_strdup(&vms->guid);

    return info;

}