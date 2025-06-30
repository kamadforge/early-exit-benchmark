import os
from abc import abstractmethod
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import torch


def get_every_but_forbidden_parameter_names(model, forbidden_layer_types):
    """
    Returns the names of the model parameters that are not inside a forbidden layer.
    """
    result = []
    for name, child in model.named_children():
        result += [
            f"{name}.{n}"
            for n in get_every_but_forbidden_parameter_names(child, forbidden_layer_types)
            if not isinstance(child, tuple(forbidden_layer_types))
        ]
    result += list(model._parameters.keys())
    return result

FORBIDDEN_LAYER_TYPES = [torch.nn.Embedding, torch.nn.LayerNorm, torch.nn.BatchNorm1d, torch.nn.BatchNorm2d]

class BaseAnalysis:
    def export(self, name):
        torch.save(self.result, os.path.join(self.rpath, name + ".pt"))

    def clean_up(self):
        for attr in self.attributes_on_gpu:
            try:
                a = getattr(self, attr)
                a.to("cpu")
                del a
            except AttributeError:
                pass
        del self
        torch.cuda.empty_cache()

    @abstractmethod
    def analysis(self):
        pass

    @abstractmethod
    def plot(self, path):
        pass


class MutualInformation(BaseAnalysis):
    def __init__(self, model, loader, modules_list, layers=None, rpath='.', MAX_REPR_SIZE=8000):
        self.model = model
        self.loader = loader
        self.names_of_layers_to_analyze = layers if layers is not None else [n for n, _ in model.named_modules()]
        self.modules_list = modules_list
        self.handels = []
        self._insert_hooks()
        self.representations = {}
        self.MAX_REPR_SIZE = MAX_REPR_SIZE
        self.rpath = rpath
        self.attributes_on_gpu = ["model"]
        self.logger = None
        self.is_able = False
        self.device = next(model.parameters()).device
        self.penalized_parameter_names = get_every_but_forbidden_parameter_names(self.model, FORBIDDEN_LAYER_TYPES)
        self.subsampling = {}
        self.eps = torch.finfo(torch.float32).eps
        self.stds = torch.tensor([0.01, 0.1, 0.25, 0.5, 0.75, 1.0, 2.0, 3.0,
                     6.0, 9.0, 12.0, 15.0, 18.0, 21.0])


    def _spectra_hook(self, name):
        def spectra_hook(model, input, output):
            if self.is_able:
                output = output.flatten(start_dim=1)
                representation_size = output.shape[1]
                if name in self.subsampling:  # czy da się lepiej? Spytać Staszka. #TODO
                    output = torch.index_select(output, 1, self.subsampling[name].to(self.device))
                elif representation_size > self.MAX_REPR_SIZE:
                    self.subsampling[name] = torch.randperm(representation_size)[:self.MAX_REPR_SIZE].sort()[0]
                    output = torch.index_select(output, 1, self.subsampling[name].to(self.device))
                
                self.representations[name] = self.representations.get(name, []) + [output]
        return spectra_hook


    def _insert_hooks(self):
        for name, module in self.model.named_modules():
            if name in self.names_of_layers_to_analyze:
                if any(isinstance(module, module_type) for module_type in self.modules_list):
                    self.handels.append(module.register_forward_hook(self._spectra_hook(name)))
          
                
    def disable(self):
        self.is_able = False
        
        
    def enable(self):
        self.is_able = True


    @torch.no_grad()
    def collect_representations(self):
        self.model.eval()
        y_true = torch.empty((0,)).to(next(self.model.parameters()).device)
        with torch.no_grad():
            for x_true, y_data in self.loader:
                _ = self.model(x_true)
                y_true = torch.cat((y_true, y_data))
        self.feats = defaultdict(list)
        self.layer_names = {}
        for i, (name, repr) in enumerate(self.representations.items()):
            self.layer_names[i] = name
            repr = torch.cat(repr, dim=0).detach()
            for y in range(len(np.unique(y_true.cpu().numpy()))):
                matches = y_true == y
                repr_y = repr[matches]
                self.feats[name].append(repr_y)
        self.model.train()
        return y_true
    
    
    def collect_weights(self):
        named_weights = {n: p.reshape(p.size(0), -1) for n, p in self.model.named_parameters() if 'weight' in n and n in self.penalized_parameter_names}
        return named_weights


    def analysis(self, step, scope, phase, tc):
        main_prefix = f'mutual_information_representations_{phase}'
        postfix = f'____{scope}____{phase}'
        self.y_true = self.collect_representations()
        print(self.layer_names)
        name_template = lambda name: f'{main_prefix}/{name}{postfix}'
        num_layers_incl_in = len(self.layer_names)
        num_classes = len(np.unique(self.y_true.cpu().numpy()))
        
        from collections import Counter
        print(Counter(self.y_true.cpu().numpy()))
    
        num_seeds = 1
        sz1 = 2000
        sz1_y = 1000
        sz2 = 400
        batch_sz = 100
        MI_X_Z_violated = 0
        
        evaluators = {}

        for mode in ["mc", "jensen"]:
            evaluators1 = {}
            evaluators2 = {}
            MI_Z_X_results = []
            logprob_results = []
            for std in self.stds:  # sorted  
                MI_Z_X_std, layer_logprob = self.compute_MI_Z_X(torch.ones(num_layers_incl_in).fill_(std), sz1, sz2, batch_sz, mode=mode)
                MI_Z_X_results.append(MI_Z_X_std)
                logprob_results.append(layer_logprob)
            MI_Z_X_results = torch.stack(MI_Z_X_results, dim=0)
            logprob_results = torch.stack(logprob_results, dim=0)

            best_std_i = torch.zeros(num_layers_incl_in, dtype=torch.long)
            
            print(self.layer_names)

            MI_constraint = -np.inf
            for curr_l in reversed(range(num_layers_incl_in)):
                std_i_sat_constraint = (MI_Z_X_results[:, curr_l] >= MI_constraint).nonzero(as_tuple=True)[0]
                assert len(std_i_sat_constraint.shape) == 1
                if std_i_sat_constraint.shape[0] == 0:
                    best_std_l_i = logprob_results[:, curr_l].argmax()
                    MI_X_Z_violated += 1
                else:
                    best_std_l_ii = logprob_results[std_i_sat_constraint, curr_l].argmax()
                    best_std_l_i = std_i_sat_constraint[best_std_l_ii]

                best_std_i[curr_l] = best_std_l_i
                MI_constraint = MI_Z_X_results[best_std_l_i, curr_l]

            best_stds = self.stds[best_std_i]
            
            print(self.layer_names)
            
            print(MI_Z_X_results.shape)
            
            
            for l in range(num_layers_incl_in):
                evaluators1[name_template(f"MI_Z_X_results_{mode}____{self.layer_names[l]}")] = MI_Z_X_results[:, l].mean()
                
            title = name_template(f"MI_Z_X_{mode}")

            MI_Z_Xy, c_l_y = self.compute_MI_Z_Xy(stds=best_stds, sz1=sz1_y, sz2=sz2, batch_sz=batch_sz, mode=mode)
            y_counts = torch.ones(num_classes) / len(self.feats[self.layer_names[0]][0])
            y_prop = y_counts / y_counts.sum()  # C
            avg_MI_Z_Xy = (MI_Z_Xy * y_prop.unsqueeze(0)).sum(dim=1)
            
            for l in range(num_layers_incl_in):
                evaluators2[name_template(f"avg_MI_Z_Xy_{mode}____{self.layer_names[l]}")] = avg_MI_Z_Xy[l]
                
            title = name_template(f"MI_Z_X|Y_{mode}")
                
            evaluators = evaluators | evaluators1 | evaluators2
        
        evaluators[f'steps/ranks_{phase}'] = step
        tc.writer.log_scalars(evaluators)

        self.representations = {}
        self.subsampling = {}
        torch.cuda.empty_cache()
        
        
    def compute_MI_Z_X(self, stds, sz1, sz2, batch_sz, mode):

        num_layers_incl_in = len(self.layer_names)

        assert stds.shape == (num_layers_incl_in,)

        log_post = torch.distributions.normal.Normal(0, stds).log_prob(torch.zeros(num_layers_incl_in))
        assert log_post.shape == (num_layers_incl_in,)

        feat_ent_bounds_i = torch.zeros(num_layers_incl_in)
        layer_logprob = torch.zeros(num_layers_incl_in)

        for l in range(num_layers_incl_in):
            feats_l = torch.cat(self.feats[self.layer_names[l]], dim=0)
            assert len(feats_l.shape) == 2
            n_s, n_feat = feats_l.shape


            ref_i = torch.tensor(np.random.choice(n_s, sz2, replace=True))
            ref_feats = feats_l[ref_i]

            distr = torch.distributions.normal.Normal(ref_feats, stds[l].item())
            
            eval_i = torch.tensor(np.random.choice(n_s, sz1, replace=True))
            eval_feats = feats_l[eval_i]
            
            log_prior = self.compute_log_prior(eval_feats, distr, sz1, sz2, batch_sz, n_feat, mode)

            feat_ent_bounds_i[l] = (log_post[l] - log_prior).mean()
            layer_logprob[l] = log_prior.mean().item()

        return feat_ent_bounds_i, layer_logprob
        
        
    def compute_MI_Z_Xy(self, stds, sz1, sz2, batch_sz, mode):
        '''
        sz1 - sample size 1 = k ?
        sz2 - sample size 2 = n_c ?
        '''
        num_layers_incl_in = len(self.layer_names)
        num_classes = len(np.unique(self.y_true.cpu().numpy()))

        feat_ent_bounds_i = torch.zeros(num_layers_incl_in, num_classes)
        c_l_y = torch.zeros(num_layers_incl_in, num_classes)
        log_post = torch.distributions.normal.Normal(0, stds).log_prob(torch.zeros(num_layers_incl_in))
        for y in range(num_classes):
            for l in range(num_layers_incl_in):
                feats_y_l = self.feats[self.layer_names[l]][y]
                assert len(feats_y_l.shape) == 2
                n_c, n_feat = feats_y_l.shape

                # MI
                print(n_c, sz2, "test")
                ref_i =  torch.tensor(np.random.choice(n_c, sz2, replace=True))
                ref_feats = feats_y_l[ref_i]

                distr = torch.distributions.normal.Normal(ref_feats, stds[l].item())
                
                eval_i = torch.tensor(np.random.choice(n_c, sz1, replace=True))
                eval_feats = feats_y_l[eval_i]
                
                log_prior = self.compute_log_prior(eval_feats, distr, sz1, sz2, batch_sz, n_feat, mode)
                feat_ent_bounds_i[l, y] = (log_post[l] - log_prior).mean()

                swapped_feats = eval_feats.clone()

                swap_i = torch.tensor(np.random.choice(sz1, sz1, replace=True))
                swap_j = torch.tensor(np.random.choice(n_feat, sz1, replace=True))

                swapped_feats[range(sz1), swap_j] = eval_feats[swap_i, swap_j]

                new_log_prior = self.compute_log_prior(swapped_feats, distr, sz1, sz2, batch_sz, n_feat, mode)
                max_diff = (new_log_prior - log_prior).abs().max()
                c_l_y[l, y] = max_diff.item()

        return feat_ent_bounds_i, c_l_y
    
    
    def compute_log_prior(self, eval_feats, distr, sz1, sz2, batch_sz, n_feat, mode):
        nb = int(np.ceil(sz1 / float(batch_sz)))
        log_priors = []
        for b_i in range(nb):
            s = (b_i * batch_sz)
            e =  min((b_i + 1) * batch_sz, sz1)
            b_sz = e - s

            eval_feats_b = eval_feats[s:e]
            eval_feats_expanded = eval_feats_b.unsqueeze(1).expand(b_sz, sz2, n_feat)
            log_prob = distr.log_prob(eval_feats_expanded)
            assert log_prob.shape == (b_sz, sz2, n_feat)

            log_prob = log_prob.sum(dim=2)

            if mode == "jensen":
                log_prior = log_prob.mean(dim=1)
            elif mode == "mc":
                log_prior =  - np.log(sz2) + torch.logsumexp(log_prob, dim=1)
            else:
                raise NotImplementedError

            log_priors.append(log_prior)

        return torch.cat(log_priors, dim=0)
    

    def plot(self, evaluators, title, tc, step):
        import io

        from PIL import Image
        fig, axs = plt.subplots(1, 1, figsize=(10, 10))
        axs.plot(list(range(len(evaluators))), list(evaluators.values()), "o-")
        axs.set_title("Rank Across Layers")
        axs.set_xlabel("Layer")
        axs.set_ylabel("Rank")
        plot_images = {title: fig}
        for name, image in plot_images.items():
            buf = io.BytesIO()
            fig.savefig(buf, format='png')
            buf.seek(0)
            image = Image.open(buf)
            image_array = np.array(image)
            image_array = image_array.transpose((2, 0, 1))
            if image_array.shape[0] == 3:
                image_array = np.concatenate((image_array, np.ones_like(image_array[0:1, ...])), axis=0)
            tc.writer.add_image(name, image_array, global_step=step)
            buf.close()
        plt.close()
