void acpi_pm1_cnt_init(ACPIREGS *ar, qemu_irq cmos_s3)

{

    ar->pm1.cnt.cmos_s3 = cmos_s3;

}
