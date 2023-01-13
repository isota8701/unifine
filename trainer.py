

lvae = lattice_vae(args)
decoder = lattice_dec(args)
latt_loss_fn = LatticeLoss(args)

for epoch in range(num_epochs):
    for batch in train_loader:
        z, (pred_num_atoms,) = lvae(batch)
        latt_loss =latt_loss_fn(latt_para, batch)
        latt_loss.backward()
        opt_l.step()
        lvae.zero_grad()

        pred_coord, energy = decoder(z, latt_para)
        coord_loss = coord_loss_fn(pred_coord, batch)
        coord_loss.backward()
        opt_c.step()
