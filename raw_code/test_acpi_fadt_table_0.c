static void test_acpi_fadt_table(test_data *data)

{

    AcpiFadtDescriptorRev1 *fadt_table = &data->fadt_table;

    uint32_t addr;



    /* FADT table comes first */

    addr = data->rsdt_tables_addr[0];

    ACPI_READ_TABLE_HEADER(fadt_table, addr);



    ACPI_READ_FIELD(fadt_table->firmware_ctrl, addr);

    ACPI_READ_FIELD(fadt_table->dsdt, addr);

    ACPI_READ_FIELD(fadt_table->model, addr);

    ACPI_READ_FIELD(fadt_table->reserved1, addr);

    ACPI_READ_FIELD(fadt_table->sci_int, addr);

    ACPI_READ_FIELD(fadt_table->smi_cmd, addr);

    ACPI_READ_FIELD(fadt_table->acpi_enable, addr);

    ACPI_READ_FIELD(fadt_table->acpi_disable, addr);

    ACPI_READ_FIELD(fadt_table->S4bios_req, addr);

    ACPI_READ_FIELD(fadt_table->reserved2, addr);

    ACPI_READ_FIELD(fadt_table->pm1a_evt_blk, addr);

    ACPI_READ_FIELD(fadt_table->pm1b_evt_blk, addr);

    ACPI_READ_FIELD(fadt_table->pm1a_cnt_blk, addr);

    ACPI_READ_FIELD(fadt_table->pm1b_cnt_blk, addr);

    ACPI_READ_FIELD(fadt_table->pm2_cnt_blk, addr);

    ACPI_READ_FIELD(fadt_table->pm_tmr_blk, addr);

    ACPI_READ_FIELD(fadt_table->gpe0_blk, addr);

    ACPI_READ_FIELD(fadt_table->gpe1_blk, addr);

    ACPI_READ_FIELD(fadt_table->pm1_evt_len, addr);

    ACPI_READ_FIELD(fadt_table->pm1_cnt_len, addr);

    ACPI_READ_FIELD(fadt_table->pm2_cnt_len, addr);

    ACPI_READ_FIELD(fadt_table->pm_tmr_len, addr);

    ACPI_READ_FIELD(fadt_table->gpe0_blk_len, addr);

    ACPI_READ_FIELD(fadt_table->gpe1_blk_len, addr);

    ACPI_READ_FIELD(fadt_table->gpe1_base, addr);

    ACPI_READ_FIELD(fadt_table->reserved3, addr);

    ACPI_READ_FIELD(fadt_table->plvl2_lat, addr);

    ACPI_READ_FIELD(fadt_table->plvl3_lat, addr);

    ACPI_READ_FIELD(fadt_table->flush_size, addr);

    ACPI_READ_FIELD(fadt_table->flush_stride, addr);

    ACPI_READ_FIELD(fadt_table->duty_offset, addr);

    ACPI_READ_FIELD(fadt_table->duty_width, addr);

    ACPI_READ_FIELD(fadt_table->day_alrm, addr);

    ACPI_READ_FIELD(fadt_table->mon_alrm, addr);

    ACPI_READ_FIELD(fadt_table->century, addr);

    ACPI_READ_FIELD(fadt_table->reserved4, addr);

    ACPI_READ_FIELD(fadt_table->reserved4a, addr);

    ACPI_READ_FIELD(fadt_table->reserved4b, addr);

    ACPI_READ_FIELD(fadt_table->flags, addr);



    ACPI_ASSERT_CMP(fadt_table->signature, "FACP");

    g_assert(!acpi_calc_checksum((uint8_t *)fadt_table, fadt_table->length));

}
