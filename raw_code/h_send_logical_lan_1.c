static target_ulong h_send_logical_lan(CPUPPCState *env, sPAPREnvironment *spapr,

                                       target_ulong opcode, target_ulong *args)

{

    target_ulong reg = args[0];

    target_ulong *bufs = args + 1;

    target_ulong continue_token = args[7];

    VIOsPAPRDevice *sdev = spapr_vio_find_by_reg(spapr->vio_bus, reg);

    VIOsPAPRVLANDevice *dev = (VIOsPAPRVLANDevice *)sdev;

    unsigned total_len;

    uint8_t *lbuf, *p;

    int i, nbufs;

    int ret;



    dprintf("H_SEND_LOGICAL_LAN(0x" TARGET_FMT_lx ", <bufs>, 0x"

            TARGET_FMT_lx ")\n", reg, continue_token);



    if (!sdev) {

        return H_PARAMETER;

    }



    dprintf("rxbufs = %d\n", dev->rx_bufs);



    if (!dev->isopen) {

        return H_DROPPED;

    }



    if (continue_token) {

        return H_HARDWARE; /* FIXME actually handle this */

    }



    total_len = 0;

    for (i = 0; i < 6; i++) {

        dprintf("   buf desc: 0x" TARGET_FMT_lx "\n", bufs[i]);

        if (!(bufs[i] & VLAN_BD_VALID)) {

            break;

        }

        total_len += VLAN_BD_LEN(bufs[i]);

    }



    nbufs = i;

    dprintf("h_send_logical_lan() %d buffers, total length 0x%x\n",

            nbufs, total_len);



    if (total_len == 0) {

        return H_SUCCESS;

    }



    if (total_len > MAX_PACKET_SIZE) {

        /* Don't let the guest force too large an allocation */

        return H_RESOURCE;

    }



    lbuf = alloca(total_len);

    p = lbuf;

    for (i = 0; i < nbufs; i++) {

        ret = spapr_tce_dma_read(sdev, VLAN_BD_ADDR(bufs[i]),

                                 p, VLAN_BD_LEN(bufs[i]));

        if (ret < 0) {

            return ret;

        }



        p += VLAN_BD_LEN(bufs[i]);

    }



    qemu_send_packet(&dev->nic->nc, lbuf, total_len);



    return H_SUCCESS;

}
