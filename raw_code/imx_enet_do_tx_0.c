static void imx_enet_do_tx(IMXFECState *s)

{

    int frame_size = 0;

    uint8_t frame[ENET_MAX_FRAME_SIZE];

    uint8_t *ptr = frame;

    uint32_t addr = s->tx_descriptor;



    while (1) {

        IMXENETBufDesc bd;

        int len;



        imx_enet_read_bd(&bd, addr);

        FEC_PRINTF("tx_bd %x flags %04x len %d data %08x option %04x "

                   "status %04x\n", addr, bd.flags, bd.length, bd.data,

                   bd.option, bd.status);

        if ((bd.flags & ENET_BD_R) == 0) {

            /* Run out of descriptors to transmit.  */

            break;

        }

        len = bd.length;

        if (frame_size + len > ENET_MAX_FRAME_SIZE) {

            len = ENET_MAX_FRAME_SIZE - frame_size;

            s->regs[ENET_EIR] |= ENET_INT_BABT;

        }

        dma_memory_read(&address_space_memory, bd.data, ptr, len);

        ptr += len;

        frame_size += len;

        if (bd.flags & ENET_BD_L) {

            if (bd.option & ENET_BD_PINS) {

                struct ip_header *ip_hd = PKT_GET_IP_HDR(frame);

                if (IP_HEADER_VERSION(ip_hd) == 4) {

                    net_checksum_calculate(frame, frame_size);

                }

            }

            if (bd.option & ENET_BD_IINS) {

                struct ip_header *ip_hd = PKT_GET_IP_HDR(frame);

                /* We compute checksum only for IPv4 frames */

                if (IP_HEADER_VERSION(ip_hd) == 4) {

                    uint16_t csum;

                    ip_hd->ip_sum = 0;

                    csum = net_raw_checksum((uint8_t *)ip_hd, sizeof(*ip_hd));

                    ip_hd->ip_sum = cpu_to_be16(csum);

                }

            }

            /* Last buffer in frame.  */

            qemu_send_packet(qemu_get_queue(s->nic), frame, len);

            ptr = frame;

            frame_size = 0;

            if (bd.option & ENET_BD_TX_INT) {

                s->regs[ENET_EIR] |= ENET_INT_TXF;

            }

        }

        if (bd.option & ENET_BD_TX_INT) {

            s->regs[ENET_EIR] |= ENET_INT_TXB;

        }

        bd.flags &= ~ENET_BD_R;

        /* Write back the modified descriptor.  */

        imx_enet_write_bd(&bd, addr);

        /* Advance to the next descriptor.  */

        if ((bd.flags & ENET_BD_W) != 0) {

            addr = s->regs[ENET_TDSR];

        } else {

            addr += sizeof(bd);

        }

    }



    s->tx_descriptor = addr;



    imx_eth_update(s);

}
