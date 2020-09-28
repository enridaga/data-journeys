# FindDependencies
import ast
#import showast
import networkx
import sys
# 
import zlib
#
from rdflib import URIRef, BNode, Literal, Namespace, Graph
from rdflib.namespace import CSVW, DC, DCAT, DCTERMS, DOAP, FOAF, ODRL2, ORG, OWL, \
                           PROF, PROV, RDF, RDFS, SDO, SH, SKOS, SOSA, SSN, TIME, \
                           VOID, XMLNS, XSD

class FindDependencies(ast.NodeVisitor):

  def __init__(self, notebook):
    self._notebook = notebook
  
  def _collectLeaves(self, v):
    bag = []
    # TODO: Expand supported types?
    if type(v) is ast.Num:
      bag.append(v.n)
    elif type(v) is ast.Str:
      bag.append(v.s)
    elif type(v) is ast.Name:
      bag.append(v.id)
    elif type(v) is ast.Constant:
      bag.append(str(v.value))
    elif type(v) is ast.List:
      bag.append(str(v.elts))
    elif type(v) is list:
      # print(v)
      for l in v:
        bag = bag + self._collectLeaves(l)
    else:
      try:
        for l in ast.iter_fields(v):
          # print(l)
          bag = bag + self._collectLeaves(l[1])
      except:
        sys.stderr.write("Skipping " + str(v) + " type: " + str(type(v)) + "\n")
    return bag

  def __variableAssign(self, symbol, scope):
    scope_index = self._scopes.index(scope) 
    symbol = symbol + "(" + str(scope_index) + ")"
    if not hasattr(self, '_vars'):
      self._vars = {}
    if symbol in self._vars.keys():
      s = self._vars[symbol]
      i = len(s)
      self._vars[symbol].append(str(symbol) + "$" + str(i)) 
      # _vars[s] = {} 
    else:
      self._vars[symbol] = []
      self._vars[symbol].append(str(symbol) + "$0")
    return self._vars[symbol][-1]

  def __variable(self, symbol, scope):
    scope_index = self._scopes.index(scope)
    symbol = str(symbol) + "(" + str(scope_index) + ")"
    if not hasattr(self, '_vars'):
      self._vars = {}
    if symbol in self._vars.keys():
      return self._vars[symbol][-1]
    # This symbol was never found before, let's map it to the notebook if in the outer scope
    if scope_index == 0:
      if not hasattr(self, '_appeared'):
        self._appeared = []
      if symbol not in self._appeared:
        self.__collect(self._notebook, "appearsIn", symbol)
        self._appeared.append(symbol)
    return symbol

  def __collect(self, source, func, target):
    if not hasattr(self, '_bag'):
      self._bag = []
    self._bag.append((source, func, target))

  def __scopeOpen(self, node):
    if not hasattr(self, '_scopes'):
      self._scopes = []
    self._scopes.append(node)

  def __scope(self, node):
    for s in reversed(self._scopes):
      if node in ast.walk(s):
        #print("Scope of" + str(node) + " is " + str(s))
        return s
    raise Exception("Scope not found!")


  def _nameFromSubscript(self, sub):
    if type(sub) is ast.Subscript:
        return self._nameFromSubscript(sub.value)
    elif type(sub) is ast.Name:
        return sub
    elif type(sub) is ast.Attribute:
        return self._nameFromSubscript(sub.value)
    elif type(sub) is ast.Call:
        return self._nameFromSubscript(sub.func)
    elif 'value' in vars(sub):
        return self._nameFromSubscript(sub.value)
    elif 'id' in vars(sub):
        return sub
    raise Exception('Unsupported Subscript ' + 'Type: ' + str(type(sub)) + str(vars(sub)))

  def _collectTargets(self, targets):
      tt = []
      for n in targets:
         if type(n) is ast.Tuple:
             tt = tt + self._collectTargets(n.elts)
         else:
             tt.append(n)
      
      return tt
  
### Visitor
  def visit_Import(self, node): # node.names
    scope = self.__scope(node)
    func = "importedBy"
    #print(vars(node), node.names)
    for alias in node.names:
      s = str(alias.name) # the module is a constant as it needs to be the same in all notebooks
      self.__collect(self._notebook, func, s) 
      o = self.__variable(str(alias.asname or alias.name), scope) # the variable, instead, needs to be different for each notebook
      self.__collect(s, "assignedFrom", o)
    self.generic_visit(node)
  
  def visit_ImportFrom(self, node): # (identifier? module, alias* names, int? level)
    scope = self.__scope(node)
    func = "importedBy"
    m = str(node.module) # the module is a constant as it needs to be the same in all notebooks
    self.__collect(self._notebook, func, m) # 
    for alias in node.names:
      s = str(alias.name)  # the module is a constant as it needs to be the same in all notebooks
      self.__collect(m, func, s)
      o = self.__variable(str(alias.asname or alias.name), scope)
      self.__collect(s, "assignedFrom", o)
    self.generic_visit(node)
  
  def visit_Module(self, node):
    self.__scopeOpen(node)
    self.generic_visit(node)
  
  def visit_ClassDef(self, node):
    self.__scopeOpen(node)
    self.generic_visit(node)
      
  def collected(self):
    return self._bag;
  
  def visit_AsyncFunctionDef(self, node):
    self.visit_FunctionDef(node)
    
  def visit_FunctionDef(self, node):
    scope = self.__scope(node)
    self.__scopeOpen(node)
    func = node.name
    c = 0
    for a in node.args.args:
      par = func + "(" + str(self._scopes.index(scope)) + ")" + '[' + str(c) + ']'
      arg = self.__variable(a.arg, node) # this node is the scope
      self.__collect(func + '[' + str(c) + ']', '_argToVar' , arg)
      c += 1
    self.generic_visit(node)

  def visit_Expr(self, node):
    scope = self.__scope(node)
    if type(node.value) is ast.Attribute and 'func' not in vars(node.value):
        # obj.property alone don't mean anything to us
        pass
    elif 'func' not in vars(node.value):
        # a var alone or obj[property] alone don't mean anything to us
        pass
    elif type(node.value.func) is ast.Attribute:
      # method expressions
      leaves = self._collectLeaves (node.value.args)
      # Subscript
      if type(node.value.func.value) is ast.Subscript:
        obj = self._nameFromSubscript(node.value.func.value).id
      elif type(node.value.func.value) is ast.Attribute:
        obj = node.value.func.value.value.id
      elif type(node.value.func.value) is ast.Call:
        obj = self._nameFromSubscript(node.value.func.value).id
      else:
        obj = node.value.func.value.id
      # if leaves is empty then method has no argument
      #if len(leaves) == 0:
      # rewrite obj with modified version
      s = self.__variable(str(obj), scope)
      o = self.__variableAssign(str(obj), scope)
      self.__collect(s, node.value.func.attr, o)
      for l in leaves:
        s = self.__variable(str(l), scope)
        #o = self.__variable(str(obj), scope)
        self.__collect(s, node.value.func.attr, o)
    else:
      # function expressions
      leaves = self._collectLeaves(node.value)
      func = leaves.pop(0)
      c = 0
      for l in leaves:
        s = self.__variable(str(l), scope)
        self.__collect(s, func, str(func) +'['+str(c)+']')
        c += 1
    self.generic_visit(node)
  
  def visit_AugAssign(self, node):
    scope = self.__scope(node)
    v = node.value
    leaves = self._collectLeaves(v)
    func = str(type(node.op).__name__)

    target = node.target
    if type(target) is ast.Subscript:
      target = self._nameFromSubscript(node.target)

    # prev var name
    o = self.__variable(str(target.id), scope)
    t = self.__variableAssign(str(target.id), scope)
    for l in leaves:
      s = self.__variable(str(l), scope)
      self.__collect(s, func, t)
      self.__collect(o, func, t) # link to old var
    self.generic_visit(node)

  def visit_Assign(self, node):
    scope = self.__scope(node)
    v = node.value
    leaves = self._collectLeaves(v)
    func = "assignedFrom"
    if type(v) is ast.BinOp:
      func = str(type(v.op).__name__)
    if type(v) is ast.Call:
      if type(v.func) is ast.Name:
        #print('function: ', v.func.id)
        func = leaves.pop(0)
      if type(v.func) is ast.Attribute:
        #print('attribute: ', v.func.value.id, v.func.attr)
        func = v.func.attr
    # if type(v) is ast.List:
    #  sys.stderr.write("Example of list", v.elts)
    # sys.stderr.write(" --- ", v ,leaves, func , "---")
    t_found = []
    for l in leaves:
      # print("leave ->", str(l))
      s = self.__variable(str(l), scope)
      # print("leave(s) ->", s)
      
      # Supporting Tuple as target. All tuple members have a dependency with the right-hand stuff
      targets = self._collectTargets(node.targets)
      
      for n in targets:
        # TODO: Support targets not having 'id'
        # Subscript, e.g. x[0][1] = "Bob"
        cloned = False
        if type(n) is ast.Subscript:
          t_id = self._nameFromSubscript(n.value).id
          cloned = True
        elif type(n) is ast.Attribute:
          # This should work for any .value structure
          t_id = self._nameFromSubscript(n.value).id 
          cloned = True
        else:
          t_id = n.id 
        if str(t_id) not in t_found:
          if cloned: # When Subscript, var depends on old one as well  
            o = self.__variable(str(t_id), scope)
          t = self.__variableAssign(str(t_id), scope)
          t_found.append(str(t_id))
          if cloned: # Generate the corresponding triple
            self.__collect(o, func, t)
        else:
          t = self.__variable(str(t_id), scope)
        # sys.stderr.write("target(s) ->", t)
        self.__collect(s, func, t)
    self.generic_visit(node)

  def visit_For(self, node):
    scope = self.__scope(node)
    src = self._collectLeaves(node.iter)
    tgt = self._collectLeaves(node.target)
    for s in src:
      s = self.__variable(s, scope)
      for t in tgt:
        t = self.__variable(t, scope)
        self.__collect(s, "iteratorOf", t)
    self.generic_visit(node)

  def visit_While(self, node):
    scope = self.__scope(node)
    #self.__collect(str(node.iter),str(node.body)) 
    # sys.stderr.write(node)
    #src = self._collectLeaves(node.iter)
    # tgt = self._collectLeaves(node.target)
    # for s in src:
    #   for t in tgt:
    #     self.__collect(s, "Iter", t)
    # atr = node.value;
    self.generic_visit(node)

  def printCollected(self):
    for t in self._bag:
      print(t[2] + " -> " + t[1] + " -> " + t[0])

  def getStringCollected(self):
    tmp_str = "digraph { \n"
    for t in self._bag:
      tmp_str = tmp_str + "\"" + t[2] + "\""+ " -> " + "\"" + t[0] + "\"" +  " [label = \"" + t[1] + "\"]"  + "\n"
    tmp_str = tmp_str + "}"
    return tmp_str
  
  def collect(self, source):
    tree = ast.parse(source)
    self.visit(tree)


def toRDF(name, digraph):
    n = name
    g = digraph
    DJ = Namespace("http://purl.org/dj/")
    K = Namespace("http://purl.org/dj/kaggle/")
    L = Namespace("http://purl.org/dj/python/lib/")
    notebook = URIRef(str(K) + n)
    Loc = Namespace(str(K) + str(n) + "#")
    #print(notebook)
    rdfg = Graph()
    rdfg.bind("rdf", RDF)
    rdfg.bind("dj", DJ)
    rdfg.bind("rdfs", RDFS)
    rdfg.bind("k", K)
    rdfg.add(( notebook, RDF.type, URIRef(str(K) + "Notebook")))
    for edge in g.edges.data('label'):
        pl = edge[2]
        sl = edge[0]
        ol = edge[1]
        # If predicate Imports, use LIB namespace on Subject
        # If the name does not have a '(' and it is not the notebook, use the lib namespace
        if sl != n and not any(specialchar in sl for specialchar in "[()]"):
            subj = URIRef(str(L) + str(zlib.adler32(str(sl).encode())))
        else:
            subj = URIRef(str(Loc) + str(zlib.adler32(str(sl).encode())))
        # If object is notebook, use Notebook entity instead
        if ol == n:
            obj = notebook
        else:
            # If the name does not have a '(' and it is not the notebook, use the lib namespace
            if ol != n and not any(specialchar in ol for specialchar in "[()]"):
                obj = URIRef(str(L) + str(zlib.adler32(str(ol).encode())))
            else:
                obj = URIRef(str(Loc) + str(zlib.adler32(str(ol).encode())))
        pred = URIRef(str(DJ) + str(pl))
        rdfg.add((subj, RDFS.label, Literal(sl)))
        rdfg.add((obj, RDFS.label, Literal(ol)))
        rdfg.add((subj, pred, obj))
    return rdfg

  # def getGraph(self):
  #   tmp_str = "digraph { \n"
  #   for t in self._bag:
  #     tmp_str = tmp_str + "\"" + self._notebook + "\""+ " -> " + "\"" + t[0] + "\"" +  " [label = \"_includesNode\"]"  + "\n"
  #     tmp_str = tmp_str + "\"" + self._notebook + "\""+ " -> " + "\"" + t[2] + "\"" +  " [label = \"_includesNode\"]"  + "\n"
  #     tmp_str = tmp_str + "\"" + t[2] + "\""+ " -> " + "\"" + t[0] + "\"" +  " [label = \"" + t[1] + "\"]"  + "\n"
  #   tmp_str = tmp_str + "}"
  #   return tmp_str