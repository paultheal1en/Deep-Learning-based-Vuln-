static void virtio_pci_modern_region_map(VirtIOPCIProxy *proxy,

                                         VirtIOPCIRegion *region,

                                         struct virtio_pci_cap *cap)

{

    memory_region_add_subregion(&proxy->modern_bar,

                                region->offset,

                                &region->mr);



    cap->cfg_type = region->type;

    cap->offset = cpu_to_le32(region->offset);

    cap->length = cpu_to_le32(memory_region_size(&region->mr));

    virtio_pci_add_mem_cap(proxy, cap);

}
