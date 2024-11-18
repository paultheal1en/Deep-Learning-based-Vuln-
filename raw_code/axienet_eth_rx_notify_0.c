static void axienet_eth_rx_notify(void *opaque)

{

    XilinxAXIEnet *s = XILINX_AXI_ENET(opaque);



    while (s->rxsize && stream_can_push(s->tx_dev, axienet_eth_rx_notify, s)) {

        size_t ret = stream_push(s->tx_dev, (void *)s->rxmem + s->rxpos,

                                 s->rxsize, s->rxapp);

        s->rxsize -= ret;

        s->rxpos += ret;

        if (!s->rxsize) {

            s->regs[R_IS] |= IS_RX_COMPLETE;

            g_free(s->rxapp);

        }

    }

    enet_update_irq(s);

}
