# Jump Your Steps (Park et al., ICLR 2025) baseline: KLUB estimate of the
# compounding decoding error and the golden-section search over intermediate
# timesteps. Adapted from https://github.com/enkeejunior1/jump-your-steps,
# restricted to the uniform graph and with the diagonal of the uniform-graph
# reverse rate excluded from the KLUB (see below).
import torch


def golden(a,b,c):
    phi = 1.618
    if b is None :
        return a + (c-a)/(1+phi)
    if b-a > c-b :
        return b-(b-a)/(1+phi),True
    else :
        return b+(c-b)/(1+phi),False


def golden_section_algo(f, a, c, b=None, next_x=golden, err=1e-4, max_iter=32):
    # find min with the golden section method
    # ref: https://www.kaggle.com/code/thomasdubail/golden-section-search
    # a < b < c & f(b) < f(c) & f(b) < f(a)
    if b is None:
        b = next_x(a,b,c)
    f_a = f(a)
    f_b = f(b)
    f_c = f(c)

    n_iter = 0
    b_list = [a,b,c]
    f_dict = {a: f_a, b: f_b, c: f_c}
    while c-a > err and max_iter > n_iter:
        x,on_left = next_x(a,b,c)
        f_x = f(x)
        f_dict[x] = f_x
        if f_x < f_b :
            b = x
            f_b = f_x
            b_list.append(b)
        elif on_left :
            a = x
            f_a = f_x
        else :
            c = x
            f_c = f_x
        n_iter += 1
    return b, b_list, f_dict


@torch.no_grad()
def klub(t1_i, t2_i, tnext_i, model, scheduler, dl, seed=42, fix_length=0, max_length=None, device=None, dtype=None):
    '''KLUB between taking one step t1 -> tnext and two steps t1 -> t2 -> tnext
    (Algorithm 1 of JYS, data-based estimate).'''
    generator1 = torch.Generator(device).manual_seed(seed)
    generator2 = torch.Generator(device).manual_seed(seed)

    t1_i = torch.tensor(t1_i).to(device, dtype)
    t2_i = torch.tensor(t2_i).to(device, dtype)
    tnext_i = torch.tensor(tnext_i).to(device, dtype)

    klub_r = 0
    klub_d = 0
    for x0, _ in dl:
        x0 = x0.to(device)
        t1 = t1_i.repeat(x0.shape[0]).to(device)
        t2 = t2_i.repeat(x0.shape[0]).to(device)
        xt1 = scheduler.add_noise(x0, t1, generator1)
        xt2 = scheduler.add_noise(x0, t2, generator2)

        # fix length (conditional generation)
        xt1[:, :fix_length] = x0[:, :fix_length]
        xt2[:, :fix_length] = x0[:, :fix_length]

        # max length
        xt1 = xt1[:, :max_length] if max_length is not None else xt1
        xt2 = xt2[:, :max_length] if max_length is not None else xt2

        # R1
        sigma_bar_1 = scheduler.sigma_bar(t1).to(device=device, dtype=dtype)
        output_1 = model(xt1, sigma_bar_1)
        rev_rate_1 = scheduler.step(output_1, xt1, t1, 0).rev_rate.float()

        # R2
        sigma_bar_2 = scheduler.sigma_bar(t2).to(device=device, dtype=dtype)
        output_2 = model(xt2, sigma_bar_2)
        rev_rate_2 = scheduler.step(output_2, xt2, t2, 0).rev_rate.float()

        # The diagonal (self-transition) entry of a uniform-graph rate row is
        # -sum(off-diagonal) by construction; it is not a jump and is
        # excluded from the KLUB.
        rev_rate_1 = rev_rate_1.scatter(-1, xt1[..., None], torch.zeros_like(rev_rate_1[..., :1]))
        rev_rate_2 = rev_rate_2.scatter(-1, xt2[..., None], torch.zeros_like(rev_rate_2[..., :1]))
        dt = t2 - tnext_i.item()
        if (rev_rate_1 <= -1e-8).any():
            print("WARNING | negative rev_rate_1:", rev_rate_1.min())
            rev_rate_1 = torch.clamp(rev_rate_1, min=1e-7)
        if (rev_rate_2 <= -1e-8).any():
            print("WARNING | negative rev_rate_2:", rev_rate_2.min())
            rev_rate_2 = torch.clamp(rev_rate_2, min=1e-7)
        token_transition_prob = dt[:, None, None] * rev_rate_2

        klub_r += (
            token_transition_prob * ((rev_rate_2 + 1e-8).log() - (rev_rate_1 + 1e-8).log())
        ).sum(dim=-1).mean()
        klub_d += 1

        assert not klub_r.isnan().any(), f"NaN in KLUB: rev_rate_1 nan? {rev_rate_1.isnan().any()}, rev_rate_2 nan? {rev_rate_2.isnan().any()}"
    return klub_r / klub_d
