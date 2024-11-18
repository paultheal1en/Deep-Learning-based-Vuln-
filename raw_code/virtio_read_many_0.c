static int virtio_read_many(ulong sector, void *load_addr, int sec_num)

{

    struct virtio_blk_outhdr out_hdr;

    u8 status;



    /* Tell the host we want to read */

    out_hdr.type = VIRTIO_BLK_T_IN;

    out_hdr.ioprio = 99;

    out_hdr.sector = sector;



    vring_send_buf(&block, &out_hdr, sizeof(out_hdr), VRING_DESC_F_NEXT);



    /* This is where we want to receive data */

    vring_send_buf(&block, load_addr, SECTOR_SIZE * sec_num,

                   VRING_DESC_F_WRITE | VRING_HIDDEN_IS_CHAIN |

                   VRING_DESC_F_NEXT);



    /* status field */

    vring_send_buf(&block, &status, sizeof(u8), VRING_DESC_F_WRITE |

                   VRING_HIDDEN_IS_CHAIN);



    /* Now we can tell the host to read */

    vring_wait_reply(&block, 0);



    drain_irqs(block.schid);



    return status;

}
