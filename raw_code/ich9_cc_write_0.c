static void ich9_cc_write(void *opaque, hwaddr addr,

                          uint64_t val, unsigned len)

{

    ICH9LPCState *lpc = (ICH9LPCState *)opaque;



    ich9_cc_addr_len(&addr, &len);

    memcpy(lpc->chip_config + addr, &val, len);

    pci_bus_fire_intx_routing_notifier(lpc->d.bus);

    ich9_cc_update(lpc);

}
