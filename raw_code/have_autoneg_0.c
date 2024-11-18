have_autoneg(E1000State *s)

{

    return (s->compat_flags & E1000_FLAG_AUTONEG) &&

           (s->phy_reg[PHY_CTRL] & MII_CR_AUTO_NEG_EN);

}
