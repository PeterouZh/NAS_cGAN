""" Network architecture visualizer using graphviz """
import os
import sys
from graphviz import Digraph

from detectron2.data import MetadataCatalog
from template_lib.utils import get_attr_kwargs
from template_lib.d2template.scripts import build_start, START_REGISTRY
from nas_cgan.models.trainer_nasgan import TrainerNASGAN


@START_REGISTRY.register()
def visualize_shared_arc(cfg, myargs):
  """ make DAG plot and save to file_path as .png """

  from template_lib.d2.data import build_cifar10_per_class

  registerd_data_name        = get_attr_kwargs(cfg, 'registerd_data_name')
  fixed_arc_file             = get_attr_kwargs(cfg, 'fixed_arc_file')
  fixed_epoch                = get_attr_kwargs(cfg, 'fixed_epoch')
  nrows                      = get_attr_kwargs(cfg, 'nrows', default=10)
  saved_file                 = get_attr_kwargs(cfg, 'saved_file')
  format                     = get_attr_kwargs(cfg, 'format')
  caption                    = get_attr_kwargs(cfg, 'caption', default=None)
  n_cells                    = get_attr_kwargs(cfg, 'n_cells')
  n_nodes_per_cell           = get_attr_kwargs(cfg, 'n_nodes_per_cell')
  conv1x1_label              = get_attr_kwargs(cfg, 'conv1x1_label')
  ops                        = get_attr_kwargs(cfg, 'ops')
  view                       = get_attr_kwargs(cfg, 'view')


  class_to_idx = MetadataCatalog.get(registerd_data_name).class_to_idx
  arcs = TrainerNASGAN._get_arc_from_file(fixed_arc_file=fixed_arc_file, fixed_epoch=fixed_epoch,
                                          nrows=nrows)
  arc = arcs[0]
  file_path = os.path.join(myargs.args.outdir, saved_file)

  edge_attr = {
    'fontsize': '40',
    'fontname': 'times'
  }
  node_attr = {
    'style'   : 'filled',
    # 'shape'   : 'circle',
    'shape': 'rect',
    'align'   : 'center',
    'fontsize': '50',
    'height'  : '1',
    'width'   : '1',
    'penwidth': '2',
    'fontname': 'times'
  }
  G = Digraph(
    filename=file_path,
    format=format,
    edge_attr=edge_attr,
    node_attr=node_attr,
    engine='dot')
  G.body.extend(['rankdir=LR'])

  n_edges_per_cell = len(arcs[0]) // n_cells
  assert n_edges_per_cell == (1 + n_nodes_per_cell) * n_nodes_per_cell // 2

  class_to_idx = list(reversed(list(class_to_idx.items())))

  plot_one_arc(file_path=file_path, G=G, n_cells=n_cells, class_name=caption, conv1x1_label=conv1x1_label,
               n_nodes_per_cell=n_nodes_per_cell,
               ops=ops, arc=arc, caption=caption, view=view)


@START_REGISTRY.register()
def visualize_every_class_arcs(cfg, myargs):
  """ make DAG plot and save to file_path as .png """

  from template_lib.d2.data import build_cifar10_per_class

  registerd_data_name        = get_attr_kwargs(cfg, 'registerd_data_name')
  fixed_arc_file             = get_attr_kwargs(cfg, 'fixed_arc_file')
  fixed_epoch                = get_attr_kwargs(cfg, 'fixed_epoch')
  saved_file                 = get_attr_kwargs(cfg, 'saved_file')
  format                     = get_attr_kwargs(cfg, 'format')
  caption                    = get_attr_kwargs(cfg, 'caption', default=None)
  n_cells                    = get_attr_kwargs(cfg, 'n_cells')
  n_nodes_per_cell           = get_attr_kwargs(cfg, 'n_nodes_per_cell')
  conv1x1_label              = get_attr_kwargs(cfg, 'conv1x1_label')
  ops                        = get_attr_kwargs(cfg, 'ops')
  view                       = get_attr_kwargs(cfg, 'view')


  class_to_idx = MetadataCatalog.get(registerd_data_name).class_to_idx
  arcs = TrainerNASGAN._get_arc_from_file(fixed_arc_file=fixed_arc_file, fixed_epoch=fixed_epoch,
                                          nrows=len(class_to_idx))


  edge_attr = {
    'fontsize': '40',
    'fontname': 'times'
  }
  node_attr = {
    'style'   : 'filled',
    # 'shape'   : 'circle',
    'shape': 'rect',
    'align'   : 'center',
    'fontsize': '50',
    'height'  : '1',
    'width'   : '1',
    'penwidth': '2',
    'fontname': 'times'
  }
  n_edges_per_cell = len(arcs[0]) // n_cells
  assert n_edges_per_cell == (1 + n_nodes_per_cell) * n_nodes_per_cell // 2

  class_to_idx = list(reversed(list(class_to_idx.items())))

  for class_name, idx in class_to_idx:
    file_path = os.path.join(myargs.args.outdir, f'{saved_file}_{idx}_{class_name}')
    G = Digraph(
      filename=file_path,
      format=format,
      edge_attr=edge_attr,
      node_attr=node_attr,
      engine='dot')
    G.body.extend(['rankdir=LR'])

    class_name = class_name.title()
    plot_one_arc(file_path=file_path, G=G, n_cells=n_cells, class_name=class_name, conv1x1_label=conv1x1_label,
                 n_nodes_per_cell=n_nodes_per_cell,
                 ops=ops, arc=arcs[idx], caption=class_name, view=view)
    pass

@START_REGISTRY.register()
def visualize_nodes(cfg, myargs):
  """ make DAG plot and save to file_path as .png """

  from template_lib.d2.data import build_cifar10_per_class

  registerd_data_name        = get_attr_kwargs(cfg, 'registerd_data_name')
  fixed_arc_file             = get_attr_kwargs(cfg, 'fixed_arc_file')
  fixed_epoch                = get_attr_kwargs(cfg, 'fixed_epoch')
  saved_file                 = get_attr_kwargs(cfg, 'saved_file')
  format                     = get_attr_kwargs(cfg, 'format')
  caption                    = get_attr_kwargs(cfg, 'caption', default=None)
  n_cells                    = get_attr_kwargs(cfg, 'n_cells')
  n_nodes_per_cell           = get_attr_kwargs(cfg, 'n_nodes_per_cell')
  conv1x1_label              = get_attr_kwargs(cfg, 'conv1x1_label')
  ops                        = get_attr_kwargs(cfg, 'ops')
  view                       = get_attr_kwargs(cfg, 'view')


  class_to_idx = MetadataCatalog.get(registerd_data_name).class_to_idx
  arcs = TrainerNASGAN._get_arc_from_file(fixed_arc_file=fixed_arc_file, fixed_epoch=fixed_epoch,
                                          nrows=len(class_to_idx))

  edge_attr = {
    'fontsize': '40',
    'fontname': 'times'
  }
  node_attr = {
    'style'   : 'filled',
    # 'shape'   : 'circle',
    'shape': 'rect',
    'align'   : 'center',
    'fontsize': '50',
    'height'  : '1',
    'width'   : '1',
    'penwidth': '2',
    'fontname': 'times'
  }

  #
  file_path = os.path.join(myargs.args.outdir, f'{saved_file}_upsample')
  G = Digraph(
    filename=file_path,
    format=format,
    edge_attr=edge_attr,
    node_attr=node_attr,
    engine='dot')
  G.body.extend(['rankdir=LR'])

  upsample_label = f"U"
  upsample_name = upsample_label
  G.node(upsample_name, upsample_label, fillcolor='yellow')
  G.render(file_path, view=view)

  file_path = os.path.join(myargs.args.outdir, f'{saved_file}_node')
  G = Digraph(
    filename=file_path,
    format=format,
    edge_attr=edge_attr,
    node_attr=node_attr,
    engine='dot')
  G.body.extend(['rankdir=LR'])
  node_name = ''
  G.node(node_name, fillcolor='lightblue')
  G.render(file_path, view=view)

  pass



def plot_one_arc(file_path, G, n_cells, class_name, conv1x1_label, n_nodes_per_cell, ops, arc, view, caption=None):

  G_ori = G
  with G.subgraph(name=f'cluster_{0}') as G:
    G.attr(color='white', penwidth='2')
    # G.attr(label=class_name)

    edge_idx = 0
    for cell_idx in range(n_cells):
      # UpSample node
      upsample_label = f"U{cell_idx}"
      upsample_name = class_name + upsample_label
      G.node(upsample_name, upsample_label, fillcolor='yellow')
      with G.subgraph(name=f'cluster_{class_name}_{cell_idx}') as g:
          g.attr(color='black', penwidth='2')
          g.attr(label=f'Cell {cell_idx}', overlap='false', fontsize='40', fontname='times')

          if cell_idx != 0:

            cell_out_label = 'Out'
            cell_out_name = class_name + f"{cell_idx-1}" + cell_out_label
            G.edge(cell_out_name, upsample_name, fillcolor="white")
          # C_in node
          cell_in_label = 'In'
          cell_in_name = class_name + f"{cell_idx - 1}" + cell_in_label
          g.node(cell_in_name, cell_in_label, fillcolor='darkseagreen2')
          G.edge(upsample_name, cell_in_name, label=conv1x1_label, fillcolor="lightgray")

          pre_nodes = [cell_in_name]
          for node_idx in range(cell_idx * n_nodes_per_cell, (cell_idx + 1) * n_nodes_per_cell):
            is_none_node = True
            # Dense connection
            for pre_node_idx in range(len(pre_nodes)):
              pre_node_name = pre_nodes[pre_node_idx]
              op_name = ops[arc[edge_idx]]
              if op_name != 'None' and pre_node_name != 'None':
                if is_none_node:
                  node_label = str(node_idx)
                  node_name = class_name + node_label
                  g.node(node_name, node_label, fillcolor='lightblue')
                g.edge(pre_node_name, node_name, label=op_name, fillcolor="black")
                is_none_node = False
              edge_idx += 1
            if is_none_node:
              pre_nodes.append('None')
            else:
              pre_nodes.append(node_name)

          # output node
          cell_out_label = 'Out'
          cell_out_name = class_name + f"{cell_idx}" + cell_out_label
          g.node(cell_out_name, cell_out_label, fillcolor='darkseagreen2')
          # Replace C_in node
          G.edge(upsample_name, cell_out_name, fillcolor="lightgray")
          for pre_node in pre_nodes[1:]:
            if pre_node != 'None':
              g.edge(pre_node, cell_out_name, fillcolor="lightgray")

    # tRGB node
    rgb = 'tRGB'
    rgb_name = class_name + rgb
    G.node(rgb_name, rgb, fillcolor='darkseagreen2')
    G.edge(cell_out_name, rgb_name, fillcolor="lightgray")


    # G.attr(label=class_name, overlap='false', fontsize='25', fontname='times')

  # add image caption
  # if caption:
  #   G_ori.attr(label=caption, overlap='false', fontsize='60', fontname='times')

  G_ori.render(file_path, view=view)
  print(f"Saved to {file_path}")

  pass


def plot(arc, cfg, **kwargs):
  """ make DAG plot and save to file_path as .png """

  format                     = get_attr_kwargs(cfg, 'format', **kwargs)
  file_path                  = get_attr_kwargs(cfg, 'file_path', **kwargs)
  caption                    = get_attr_kwargs(cfg, 'caption', default=None, **kwargs)
  n_cells                    = get_attr_kwargs(cfg, 'n_cells', **kwargs)
  n_nodes_per_cell           = get_attr_kwargs(cfg, 'n_nodes_per_cell', **kwargs)
  conv1x1_label              = get_attr_kwargs(cfg, 'conv1x1_label', **kwargs)
  ops                        = get_attr_kwargs(cfg, 'ops', **kwargs)
  view                       = get_attr_kwargs(cfg, 'view', **kwargs)

  edge_attr = {
    'fontsize': '15',
    'fontname': 'times'
  }
  node_attr = {
    'style'   : 'filled',
    # 'shape'   : 'circle',
    'shape': 'rect',
    'align'   : 'center',
    'fontsize': '15',
    'height'  : '0.5',
    'width'   : '0.5',
    'penwidth': '2',
    'fontname': 'times'
  }
  G = Digraph(
    format=format,
    edge_attr=edge_attr,
    node_attr=node_attr,
    engine='dot')
  G.body.extend(['rankdir=LR'])

  # intermediate nodes
  n_edges_per_cell = len(arc) // n_cells
  assert n_edges_per_cell == (1 + n_nodes_per_cell) * n_nodes_per_cell // 2

  edge_idx = 0
  for cell_idx in range(n_cells):
    # UpSample node
    G.node(f"U{cell_idx}", fillcolor='yellow')
    with G.subgraph(name=f'cluster_{cell_idx}') as g:
        g.attr(color='black')
        g.attr(label=f'Cell {cell_idx}', overlap='false', fontsize='20', fontname='times')

        if cell_idx != 0:
          G.edge(f"C{cell_idx-1}_{{out}}", f"U{cell_idx}", fillcolor="white")
        # C_in node
        g.node(f"C{cell_idx}_{{in}}", fillcolor='darkseagreen2')
        G.edge(f"U{cell_idx}", f"C{cell_idx}_{{in}}", label=conv1x1_label, fillcolor="lightgray")

        pre_nodes = [f"C{cell_idx}_{{in}}"]
        for node_idx in range(cell_idx * n_nodes_per_cell, (cell_idx + 1) * n_nodes_per_cell):
          is_none_node = True
          # Dense connection
          for pre_node_idx in range(len(pre_nodes)):
            pre_node_name = pre_nodes[pre_node_idx]
            op_name = ops[arc[edge_idx]]
            if op_name != 'None' and pre_node_name != 'None':
              if is_none_node:
                g.node(str(node_idx), fillcolor='lightblue')
              g.edge(pre_node_name, str(node_idx), label=op_name, fillcolor="black")
              is_none_node = False
            edge_idx += 1
          if is_none_node:
            pre_nodes.append('None')
          else:
            pre_nodes.append(str(node_idx))

        # output node
        g.node(f"C{cell_idx}_{{out}}", fillcolor='darkseagreen2')
        # Replace C_in node
        G.edge(f"U{cell_idx}", f"C{cell_idx}_{{out}}", fillcolor="lightgray")
        for pre_node in pre_nodes[1:]:
          if pre_node != 'None':
            g.edge(pre_node, f"C{cell_idx}_{{out}}", fillcolor="lightgray")

  #
  # # output node
  # g.node("c_{k}", fillcolor='palegoldenrod')
  # for i in range(n_nodes):
  #     g.edge(str(i), "c_{k}", fillcolor="gray")

  # add image caption
  if caption:
    G.attr(label=caption, overlap='false', fontsize='25', fontname='times')

  G.render(file_path, view=view)


def plot_merge_class_arcs(arcs, class_to_idx, cfg, **kwargs):
  """ make DAG plot and save to file_path as .png """

  format                     = get_attr_kwargs(cfg, 'format', **kwargs)
  file_path                  = get_attr_kwargs(cfg, 'file_path', **kwargs)
  caption                    = get_attr_kwargs(cfg, 'caption', default=None, **kwargs)
  n_cells                    = get_attr_kwargs(cfg, 'n_cells', **kwargs)
  n_nodes_per_cell           = get_attr_kwargs(cfg, 'n_nodes_per_cell', **kwargs)
  conv1x1_label              = get_attr_kwargs(cfg, 'conv1x1_label', **kwargs)
  ops                        = get_attr_kwargs(cfg, 'ops', **kwargs)
  view                       = get_attr_kwargs(cfg, 'view', **kwargs)

  edge_attr = {
    'fontsize': '15',
    'fontname': 'times'
  }
  node_attr = {
    'style'   : 'filled',
    # 'shape'   : 'circle',
    'shape': 'rect',
    'align'   : 'center',
    'fontsize': '15',
    'height'  : '0.5',
    'width'   : '0.5',
    'penwidth': '2',
    'fontname': 'times'
  }
  G = Digraph(
    filename=file_path,
    format=format,
    edge_attr=edge_attr,
    node_attr=node_attr,
    engine='dot')
  G.body.extend(['rankdir=LR'])

  n_edges_per_cell = len(arcs[0]) // n_cells
  assert n_edges_per_cell == (1 + n_nodes_per_cell) * n_nodes_per_cell // 2

  # for cell_idx in range(n_cells):
  #   with G.subgraph(name='subgraph_cell_idx' + str(cell_idx)) as s:
  #     s.attr(rank='same')
  #     for class_name, idx in class_to_idx.items():
  #       class_name = class_name.title()
  #       # UpSample node
  #       upsample_label = f"U{cell_idx}"
  #       upsample_name = class_name + upsample_label
  #       s.node(upsample_name, upsample_label, fillcolor='yellow')

  class_to_idx = list(reversed(list(class_to_idx.items())))
  G_ori = G
  for class_name, idx in class_to_idx:
    class_name = class_name.title()
    G = G_ori
    with G.subgraph(name=f'cluster_{idx}') as G:
      G.attr(color='blue', penwidth='3')
      # G.attr(label=class_name)
      G.attr(label=class_name, overlap='false', fontsize='25', fontname='times')

      edge_idx = 0
      for cell_idx in range(n_cells):
        # UpSample node
        upsample_label = f"U{cell_idx}"
        upsample_name = class_name + upsample_label
        G.node(upsample_name, upsample_label, fillcolor='yellow')
        with G.subgraph(name=f'cluster_{class_name}_{cell_idx}') as g:
            g.attr(color='black', penwidth='2')
            g.attr(label=f'Cell {cell_idx}', overlap='false', fontsize='20', fontname='times')

            if cell_idx != 0:

              cell_out_label = 'Out'
              cell_out_name = class_name + f"{cell_idx-1}" + cell_out_label
              G.edge(cell_out_name, upsample_name, fillcolor="white")
            # C_in node
            cell_in_label = 'In'
            cell_in_name = class_name + f"{cell_idx - 1}" + cell_in_label
            g.node(cell_in_name, cell_in_label, fillcolor='darkseagreen2')
            G.edge(upsample_name, cell_in_name, label=conv1x1_label, fillcolor="lightgray")

            pre_nodes = [cell_in_name]
            for node_idx in range(cell_idx * n_nodes_per_cell, (cell_idx + 1) * n_nodes_per_cell):
              is_none_node = True
              # Dense connection
              for pre_node_idx in range(len(pre_nodes)):
                pre_node_name = pre_nodes[pre_node_idx]
                op_name = ops[arcs[idx][edge_idx]]
                if op_name != 'None' and pre_node_name != 'None':
                  if is_none_node:
                    node_label = str(node_idx)
                    node_name = class_name + node_label
                    g.node(node_name, node_label, fillcolor='lightblue')
                  g.edge(pre_node_name, node_name, label=op_name, fillcolor="black")
                  is_none_node = False
                edge_idx += 1
              if is_none_node:
                pre_nodes.append('None')
              else:
                pre_nodes.append(node_name)

            # output node
            cell_out_label = 'Out'
            cell_out_name = class_name + f"{cell_idx}" + cell_out_label
            g.node(cell_out_name, cell_out_label, fillcolor='darkseagreen2')
            # Replace C_in node
            G.edge(upsample_name, cell_out_name, fillcolor="lightgray")
            for pre_node in pre_nodes[1:]:
              if pre_node != 'None':
                g.edge(pre_node, cell_out_name, fillcolor="lightgray")


  # add image caption
  if caption:
    G_ori.attr(label=caption, overlap='false', fontsize='25', fontname='times')

  if view:
    G_ori.view()
  else:
    G_ori.save()


@START_REGISTRY.register()
def visualize_class_arcs(cfg, myargs):
  # Register dataset
  from template_lib.d2.data import build_cifar10_per_class

  registerd_data_name = get_attr_kwargs(cfg, 'registerd_data_name')
  fixed_arc_file = get_attr_kwargs(cfg, 'fixed_arc_file')
  fixed_epoch = get_attr_kwargs(cfg, 'fixed_epoch')

  class_to_idx = MetadataCatalog.get(registerd_data_name).class_to_idx
  arcs = TrainerNASGAN._get_arc_from_file(fixed_arc_file=fixed_arc_file, fixed_epoch=fixed_epoch,
                                          nrows=len(class_to_idx))
  for class_name, idx in class_to_idx.items():
    class_name = class_name.title()
    plot(arc=arcs[idx], file_path=os.path.join(cfg.outdir, f'{idx}_{class_name}'), cfg=cfg, caption=class_name)
  pass


@START_REGISTRY.register()
def visualize_class_arcs_merge(cfg, myargs):
  # Register dataset
  from template_lib.d2.data import build_cifar10_per_class

  registerd_data_name = get_attr_kwargs(cfg, 'registerd_data_name')
  fixed_arc_file = get_attr_kwargs(cfg, 'fixed_arc_file')
  fixed_epoch = get_attr_kwargs(cfg, 'fixed_epoch')

  class_to_idx = MetadataCatalog.get(registerd_data_name).class_to_idx
  arcs = TrainerNASGAN._get_arc_from_file(fixed_arc_file=fixed_arc_file, fixed_epoch=fixed_epoch,
                                          nrows=len(class_to_idx))

  plot_merge_class_arcs(arcs=arcs, class_to_idx=class_to_idx,
                        file_path=os.path.join(cfg.outdir, 'merged_class_arcs'), cfg=cfg)
  pass



@START_REGISTRY.register()
def visualize_supernet(cfg, myargs):
  # Register dataset
  from template_lib.d2.data import build_cifar10_per_class

  registerd_data_name = get_attr_kwargs(cfg, 'registerd_data_name')
  fixed_arc_file = get_attr_kwargs(cfg, 'fixed_arc_file')
  fixed_epoch = get_attr_kwargs(cfg, 'fixed_epoch')

  class_to_idx = MetadataCatalog.get(registerd_data_name).class_to_idx
  arcs = TrainerNASGAN._get_arc_from_file(fixed_arc_file=fixed_arc_file, fixed_epoch=fixed_epoch,
                                          nrows=len(class_to_idx))
  plot(arc=arcs[0], file_path=os.path.join(cfg.outdir, 'supernet'), cfg=cfg, caption='Super Network')
  pass


def run(argv_str=None):
  from template_lib.utils.config import parse_args_and_setup_myargs, config2args
  from template_lib.utils.modelarts_utils import prepare_dataset
  run_script = os.path.relpath(__file__, os.getcwd())
  args1, myargs, _ = parse_args_and_setup_myargs(argv_str, run_script=run_script, start_tb=False)
  myargs.args = args1
  myargs.config = getattr(myargs.config, args1.command)

  args = config2args(myargs.config, args1)

  build_start(cfg=args, myargs=myargs)
  pass


if __name__ == '__main__':
  run()
