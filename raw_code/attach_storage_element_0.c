static void attach_storage_element(SCLPDevice *sclp, SCCB *sccb,

                                   uint16_t element)

{

    int i, assigned, subincrement_id;

    AttachStorageElement *attach_info = (AttachStorageElement *) sccb;

    sclpMemoryHotplugDev *mhd = get_sclp_memory_hotplug_dev();



    assert(mhd);



    if (element != 1) {

        sccb->h.response_code = cpu_to_be16(SCLP_RC_INVALID_SCLP_COMMAND);

        return;

    }



    assigned = mhd->standby_mem_size >> mhd->increment_size;

    attach_info->assigned = cpu_to_be16(assigned);

    subincrement_id = ((ram_size >> mhd->increment_size) << 16)

                      + SCLP_STARTING_SUBINCREMENT_ID;

    for (i = 0; i < assigned; i++) {

        attach_info->entries[i] = cpu_to_be32(subincrement_id);

        subincrement_id += SCLP_INCREMENT_UNIT;

    }

    sccb->h.response_code = cpu_to_be16(SCLP_RC_NORMAL_COMPLETION);

}
