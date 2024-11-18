static void vmxnet3_update_vlan_filters(VMXNET3State *s)

{

    int i;



    /* Copy configuration from shared memory */

    VMXNET3_READ_DRV_SHARED(s->drv_shmem,

                            devRead.rxFilterConf.vfTable,

                            s->vlan_table,

                            sizeof(s->vlan_table));



    /* Invert byte order when needed */

    for (i = 0; i < ARRAY_SIZE(s->vlan_table); i++) {

        s->vlan_table[i] = le32_to_cpu(s->vlan_table[i]);

    }



    /* Dump configuration for debugging purposes */

    VMW_CFPRN("Configured VLANs:");

    for (i = 0; i < sizeof(s->vlan_table) * 8; i++) {

        if (VMXNET3_VFTABLE_ENTRY_IS_SET(s->vlan_table, i)) {

            VMW_CFPRN("\tVLAN %d is present", i);

        }

    }

}
