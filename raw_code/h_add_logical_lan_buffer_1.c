static target_ulong h_add_logical_lan_buffer(CPUPPCState *env,

                                             sPAPREnvironment *spapr,

                                             target_ulong opcode,

                                             target_ulong *args)

{

    target_ulong reg = args[0];

    target_ulong buf = args[1];

    VIOsPAPRDevice *sdev = spapr_vio_find_by_reg(spapr->vio_bus, reg);

    VIOsPAPRVLANDevice *dev = (VIOsPAPRVLANDevice *)sdev;

    vlan_bd_t bd;



    dprintf("H_ADD_LOGICAL_LAN_BUFFER(0x" TARGET_FMT_lx

            ", 0x" TARGET_FMT_lx ")\n", reg, buf);



    if (!sdev) {

        hcall_dprintf("Bad device\n");

        return H_PARAMETER;

    }



    if ((check_bd(dev, buf, 4) < 0)

        || (VLAN_BD_LEN(buf) < 16)) {

        hcall_dprintf("Bad buffer enqueued\n");

        return H_PARAMETER;

    }



    if (!dev->isopen || dev->rx_bufs >= VLAN_MAX_BUFS) {

        return H_RESOURCE;

    }



    do {

        dev->add_buf_ptr += 8;

        if (dev->add_buf_ptr >= SPAPR_VIO_TCE_PAGE_SIZE) {

            dev->add_buf_ptr = VLAN_RX_BDS_OFF;

        }



        bd = ldq_tce(sdev, dev->buf_list + dev->add_buf_ptr);

    } while (bd & VLAN_BD_VALID);



    stq_tce(sdev, dev->buf_list + dev->add_buf_ptr, buf);



    dev->rx_bufs++;



    dprintf("h_add_logical_lan_buffer():  Added buf  ptr=%d  rx_bufs=%d"

            " bd=0x%016llx\n", dev->add_buf_ptr, dev->rx_bufs,

            (unsigned long long)buf);



    return H_SUCCESS;

}
