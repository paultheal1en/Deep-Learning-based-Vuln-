int rx_produce(World *world, uint32_t pport,

               const struct iovec *iov, int iovcnt, uint8_t copy_to_cpu)

{

    Rocker *r = world_rocker(world);

    PCIDevice *dev = (PCIDevice *)r;

    DescRing *ring = rocker_get_rx_ring_by_pport(r, pport);

    DescInfo *info = desc_ring_fetch_desc(ring);

    char *data;

    size_t data_size = iov_size(iov, iovcnt);

    char *buf;

    uint16_t rx_flags = 0;

    uint16_t rx_csum = 0;

    size_t tlv_size;

    RockerTlv *tlvs[ROCKER_TLV_RX_MAX + 1];

    hwaddr frag_addr;

    uint16_t frag_max_len;

    int pos;

    int err;



    if (!info) {

        return -ROCKER_ENOBUFS;

    }



    buf = desc_get_buf(info, false);

    if (!buf) {

        err = -ROCKER_ENXIO;

        goto out;

    }

    rocker_tlv_parse(tlvs, ROCKER_TLV_RX_MAX, buf, desc_tlv_size(info));



    if (!tlvs[ROCKER_TLV_RX_FRAG_ADDR] ||

        !tlvs[ROCKER_TLV_RX_FRAG_MAX_LEN]) {

        err = -ROCKER_EINVAL;

        goto out;

    }



    frag_addr = rocker_tlv_get_le64(tlvs[ROCKER_TLV_RX_FRAG_ADDR]);

    frag_max_len = rocker_tlv_get_le16(tlvs[ROCKER_TLV_RX_FRAG_MAX_LEN]);



    if (data_size > frag_max_len) {

        err = -ROCKER_EMSGSIZE;

        goto out;

    }



    if (copy_to_cpu) {

        rx_flags |= ROCKER_RX_FLAGS_FWD_OFFLOAD;

    }



    /* XXX calc rx flags/csum */



    tlv_size = rocker_tlv_total_size(sizeof(uint16_t)) + /* flags */

               rocker_tlv_total_size(sizeof(uint16_t)) + /* scum */

               rocker_tlv_total_size(sizeof(uint64_t)) + /* frag addr */

               rocker_tlv_total_size(sizeof(uint16_t)) + /* frag max len */

               rocker_tlv_total_size(sizeof(uint16_t));  /* frag len */



    if (tlv_size > desc_buf_size(info)) {

        err = -ROCKER_EMSGSIZE;

        goto out;

    }



    /* TODO:

     * iov dma write can be optimized in similar way e1000 does it in

     * e1000_receive_iov. But maybe if would make sense to introduce

     * generic helper iov_dma_write.

     */



    data = g_malloc(data_size);

    if (!data) {

        err = -ROCKER_ENOMEM;

        goto out;

    }

    iov_to_buf(iov, iovcnt, 0, data, data_size);

    pci_dma_write(dev, frag_addr, data, data_size);

    g_free(data);



    pos = 0;

    rocker_tlv_put_le16(buf, &pos, ROCKER_TLV_RX_FLAGS, rx_flags);

    rocker_tlv_put_le16(buf, &pos, ROCKER_TLV_RX_CSUM, rx_csum);

    rocker_tlv_put_le64(buf, &pos, ROCKER_TLV_RX_FRAG_ADDR, frag_addr);

    rocker_tlv_put_le16(buf, &pos, ROCKER_TLV_RX_FRAG_MAX_LEN, frag_max_len);

    rocker_tlv_put_le16(buf, &pos, ROCKER_TLV_RX_FRAG_LEN, data_size);



    err = desc_set_buf(info, tlv_size);



out:

    if (desc_ring_post_desc(ring, err)) {

        rocker_msix_irq(r, ROCKER_MSIX_VEC_RX(pport - 1));

    }



    return err;

}
