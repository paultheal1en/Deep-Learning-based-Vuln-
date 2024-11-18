uint32_t vfio_pci_read_config(PCIDevice *pdev, uint32_t addr, int len)

{

    VFIOPCIDevice *vdev = DO_UPCAST(VFIOPCIDevice, pdev, pdev);

    uint32_t emu_bits = 0, emu_val = 0, phys_val = 0, val;



    memcpy(&emu_bits, vdev->emulated_config_bits + addr, len);

    emu_bits = le32_to_cpu(emu_bits);



    if (emu_bits) {

        emu_val = pci_default_read_config(pdev, addr, len);

    }



    if (~emu_bits & (0xffffffffU >> (32 - len * 8))) {

        ssize_t ret;



        ret = pread(vdev->vbasedev.fd, &phys_val, len,

                    vdev->config_offset + addr);

        if (ret != len) {

            error_report("%s(%04x:%02x:%02x.%x, 0x%x, 0x%x) failed: %m",

                         __func__, vdev->host.domain, vdev->host.bus,

                         vdev->host.slot, vdev->host.function, addr, len);

            return -errno;

        }

        phys_val = le32_to_cpu(phys_val);

    }



    val = (emu_val & emu_bits) | (phys_val & ~emu_bits);



    trace_vfio_pci_read_config(vdev->vbasedev.name, addr, len, val);



    return val;

}
