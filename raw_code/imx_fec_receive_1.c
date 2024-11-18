static ssize_t imx_fec_receive(NetClientState *nc, const uint8_t *buf,

                               size_t len)

{

    IMXFECState *s = IMX_FEC(qemu_get_nic_opaque(nc));

    IMXFECBufDesc bd;

    uint32_t flags = 0;

    uint32_t addr;

    uint32_t crc;

    uint32_t buf_addr;

    uint8_t *crc_ptr;

    unsigned int buf_len;

    size_t size = len;



    FEC_PRINTF("len %d\n", (int)size);



    if (!s->regs[ENET_RDAR]) {

        qemu_log_mask(LOG_GUEST_ERROR, "[%s]%s: Unexpected packet\n",

                      TYPE_IMX_FEC, __func__);

        return 0;

    }



    /* 4 bytes for the CRC.  */

    size += 4;

    crc = cpu_to_be32(crc32(~0, buf, size));

    crc_ptr = (uint8_t *) &crc;



    /* Huge frames are truncated.  */

    if (size > ENET_MAX_FRAME_SIZE) {

        size = ENET_MAX_FRAME_SIZE;

        flags |= ENET_BD_TR | ENET_BD_LG;

    }



    /* Frames larger than the user limit just set error flags.  */

    if (size > (s->regs[ENET_RCR] >> 16)) {

        flags |= ENET_BD_LG;

    }



    addr = s->rx_descriptor;

    while (size > 0) {

        imx_fec_read_bd(&bd, addr);

        if ((bd.flags & ENET_BD_E) == 0) {

            /* No descriptors available.  Bail out.  */

            /*

             * FIXME: This is wrong. We should probably either

             * save the remainder for when more RX buffers are

             * available, or flag an error.

             */

            qemu_log_mask(LOG_GUEST_ERROR, "[%s]%s: Lost end of frame\n",

                          TYPE_IMX_FEC, __func__);

            break;

        }

        buf_len = (size <= s->regs[ENET_MRBR]) ? size : s->regs[ENET_MRBR];

        bd.length = buf_len;

        size -= buf_len;



        FEC_PRINTF("rx_bd 0x%x length %d\n", addr, bd.length);



        /* The last 4 bytes are the CRC.  */

        if (size < 4) {

            buf_len += size - 4;

        }

        buf_addr = bd.data;

        dma_memory_write(&address_space_memory, buf_addr, buf, buf_len);

        buf += buf_len;

        if (size < 4) {

            dma_memory_write(&address_space_memory, buf_addr + buf_len,

                             crc_ptr, 4 - size);

            crc_ptr += 4 - size;

        }

        bd.flags &= ~ENET_BD_E;

        if (size == 0) {

            /* Last buffer in frame.  */

            bd.flags |= flags | ENET_BD_L;

            FEC_PRINTF("rx frame flags %04x\n", bd.flags);

            s->regs[ENET_EIR] |= ENET_INT_RXF;

        } else {

            s->regs[ENET_EIR] |= ENET_INT_RXB;

        }

        imx_fec_write_bd(&bd, addr);

        /* Advance to the next descriptor.  */

        if ((bd.flags & ENET_BD_W) != 0) {

            addr = s->regs[ENET_RDSR];

        } else {

            addr += sizeof(bd);

        }

    }

    s->rx_descriptor = addr;

    imx_eth_enable_rx(s);

    imx_eth_update(s);

    return len;

}
