static int tx_consume(Rocker *r, DescInfo *info)

{

    PCIDevice *dev = PCI_DEVICE(r);

    char *buf = desc_get_buf(info, true);

    RockerTlv *tlv_frag;

    RockerTlv *tlvs[ROCKER_TLV_TX_MAX + 1];

    struct iovec iov[ROCKER_TX_FRAGS_MAX] = { { 0, }, };

    uint32_t pport;

    uint32_t port;

    uint16_t tx_offload = ROCKER_TX_OFFLOAD_NONE;

    uint16_t tx_l3_csum_off = 0;

    uint16_t tx_tso_mss = 0;

    uint16_t tx_tso_hdr_len = 0;

    int iovcnt = 0;

    int err = ROCKER_OK;

    int rem;

    int i;



    if (!buf) {

        return -ROCKER_ENXIO;

    }



    rocker_tlv_parse(tlvs, ROCKER_TLV_TX_MAX, buf, desc_tlv_size(info));



    if (!tlvs[ROCKER_TLV_TX_FRAGS]) {

        return -ROCKER_EINVAL;

    }



    pport = rocker_get_pport_by_tx_ring(r, desc_get_ring(info));

    if (!fp_port_from_pport(pport, &port)) {

        return -ROCKER_EINVAL;

    }



    if (tlvs[ROCKER_TLV_TX_OFFLOAD]) {

        tx_offload = rocker_tlv_get_u8(tlvs[ROCKER_TLV_TX_OFFLOAD]);

    }



    switch (tx_offload) {

    case ROCKER_TX_OFFLOAD_L3_CSUM:

        if (!tlvs[ROCKER_TLV_TX_L3_CSUM_OFF]) {

            return -ROCKER_EINVAL;

        }

        break;

    case ROCKER_TX_OFFLOAD_TSO:

        if (!tlvs[ROCKER_TLV_TX_TSO_MSS] ||

            !tlvs[ROCKER_TLV_TX_TSO_HDR_LEN]) {

            return -ROCKER_EINVAL;

        }

        break;

    }



    if (tlvs[ROCKER_TLV_TX_L3_CSUM_OFF]) {

        tx_l3_csum_off = rocker_tlv_get_le16(tlvs[ROCKER_TLV_TX_L3_CSUM_OFF]);

    }



    if (tlvs[ROCKER_TLV_TX_TSO_MSS]) {

        tx_tso_mss = rocker_tlv_get_le16(tlvs[ROCKER_TLV_TX_TSO_MSS]);

    }



    if (tlvs[ROCKER_TLV_TX_TSO_HDR_LEN]) {

        tx_tso_hdr_len = rocker_tlv_get_le16(tlvs[ROCKER_TLV_TX_TSO_HDR_LEN]);

    }



    rocker_tlv_for_each_nested(tlv_frag, tlvs[ROCKER_TLV_TX_FRAGS], rem) {

        hwaddr frag_addr;

        uint16_t frag_len;



        if (rocker_tlv_type(tlv_frag) != ROCKER_TLV_TX_FRAG) {

            err = -ROCKER_EINVAL;

            goto err_bad_attr;

        }



        rocker_tlv_parse_nested(tlvs, ROCKER_TLV_TX_FRAG_ATTR_MAX, tlv_frag);



        if (!tlvs[ROCKER_TLV_TX_FRAG_ATTR_ADDR] ||

            !tlvs[ROCKER_TLV_TX_FRAG_ATTR_LEN]) {

            err = -ROCKER_EINVAL;

            goto err_bad_attr;

        }



        frag_addr = rocker_tlv_get_le64(tlvs[ROCKER_TLV_TX_FRAG_ATTR_ADDR]);

        frag_len = rocker_tlv_get_le16(tlvs[ROCKER_TLV_TX_FRAG_ATTR_LEN]);



        iov[iovcnt].iov_len = frag_len;

        iov[iovcnt].iov_base = g_malloc(frag_len);

        if (!iov[iovcnt].iov_base) {

            err = -ROCKER_ENOMEM;

            goto err_no_mem;

        }



        if (pci_dma_read(dev, frag_addr, iov[iovcnt].iov_base,

                     iov[iovcnt].iov_len)) {

            err = -ROCKER_ENXIO;

            goto err_bad_io;

        }



        if (++iovcnt > ROCKER_TX_FRAGS_MAX) {

            goto err_too_many_frags;

        }

    }



    if (iovcnt) {

        /* XXX perform Tx offloads */

        /* XXX   silence compiler for now */

        tx_l3_csum_off += tx_tso_mss = tx_tso_hdr_len = 0;

    }



    err = fp_port_eg(r->fp_port[port], iov, iovcnt);



err_too_many_frags:

err_bad_io:

err_no_mem:

err_bad_attr:

    for (i = 0; i < ROCKER_TX_FRAGS_MAX; i++) {

        if (iov[i].iov_base) {

            g_free(iov[i].iov_base);

        }

    }



    return err;

}
