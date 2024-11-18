static void pci_vpb_unmap(SysBusDevice *dev, target_phys_addr_t base)

{

    PCIVPBState *s = (PCIVPBState *)dev;

    /* Selfconfig area.  */

    memory_region_del_subregion(get_system_memory(), &s->mem_config);

    /* Normal config area.  */

    memory_region_del_subregion(get_system_memory(), &s->mem_config2);



    if (s->realview) {

        /* IO memory area.  */

        memory_region_del_subregion(get_system_memory(), &s->isa);

    }

}
