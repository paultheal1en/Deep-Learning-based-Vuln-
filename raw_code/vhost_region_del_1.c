static void vhost_region_del(MemoryListener *listener,

                             MemoryRegionSection *section)

{

    struct vhost_dev *dev = container_of(listener, struct vhost_dev,

                                         memory_listener);

    int i;



    vhost_set_memory(listener, section, false);

    for (i = 0; i < dev->n_mem_sections; ++i) {

        if (dev->mem_sections[i].offset_within_address_space

            == section->offset_within_address_space) {

            --dev->n_mem_sections;

            memmove(&dev->mem_sections[i], &dev->mem_sections[i+1],

                    dev->n_mem_sections - i);

            break;

        }

    }

}
