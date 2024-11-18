static void imx_fec_do_tx(IMXFECState *s)

{

    int frame_size = 0;

    uint8_t frame[ENET_MAX_FRAME_SIZE];

    uint8_t *ptr = frame;

    uint32_t addr = s->tx_descriptor;



    while (1) {

        IMXFECBufDesc bd;

        int len;



        imx_fec_read_bd(&bd, addr);

        FEC_PRINTF("tx_bd %x flags %04x len %d data %08x\n",

                   addr, bd.flags, bd.length, bd.data);

        if ((bd.flags & ENET_BD_R) == 0) {

            /* Run out of descriptors to transmit.  */

            FEC_PRINTF("tx_bd ran out of descriptors to transmit\n");

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

            /* Last buffer in frame.  */

            qemu_send_packet(qemu_get_queue(s->nic), frame, frame_size);

            ptr = frame;

            frame_size = 0;

            s->regs[ENET_EIR] |= ENET_INT_TXF;

        }

        s->regs[ENET_EIR] |= ENET_INT_TXB;

        bd.flags &= ~ENET_BD_R;

        /* Write back the modified descriptor.  */

        imx_fec_write_bd(&bd, addr);

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
