# FindDependencies
import ast
#import showast

class FindDependencies(ast.NodeVisitor):

  def _collectLeaves(self, v):
    bag = []
    # TODO: Expand supported types?
    if type(v) is ast.Num:
      bag.append(v.n)
    elif type(v) is ast.Str:
      bag.append(v.s)
    elif type(v) is ast.Name:
      bag.append(v.id)
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
        print("Skipping " + str(v) + " type: " + str(type(v)))
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
    elif 'value' in vars(sub):
        return self._nameFromSubscript(sub.value)
    elif 'id' in vars(sub):
        return sub
    raise Exception('Unsupported Subscript ' + 'Type: ' + str(type(sub)) + str(vars(sub)))

### Visitor

  def visit_Module(self, node):
    self.__scopeOpen(node)
    self.generic_visit(node)

  def collected(self):
    return self._bag;
  
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
    func = "eq"
    if type(v) is ast.BinOp:
      func = str(type(v.op).__name__)
    if type(v) is ast.Call:
      if type(v.func) is ast.Name:
        #print('function: ', v.func.id)
        func = leaves.pop(0)
      if type(v.func) is ast.Attribute:
        #print('attribute: ', v.func.value.id, v.func.attr)
        func = v.func.attr
    #if type(v) is ast.List:
    #  print("Example of list", v.elts)
    t_found = []
    for l in leaves:
      #print("leave ->", str(l))
      s = self.__variable(str(l), scope)
      #print("leave(s) ->", s)
      
      # Supporting Tuple as target. All tuple members have a dependency with the right-hand stuff
      targets=[]
      for n in node.targets:
         if type(n) is ast.Tuple:
             for u in n.elts:
                 targets.append(u)
         else:
             targets.append(n)

      for n in targets:
        # TODO: Support targets not having 'id'
        # Subscript, e.g. x[0][1] = "Bob"
        if type(n) is ast.Subscript:
          t_id = self._nameFromSubscript(n.value).id
        elif type(n) is ast.Tuple:
          print(vars(n))
          t_id = n.id # BROKEN
        else:
          t_id = n.id
        if str(t_id) not in t_found:
          if type(n) is ast.Subscript: # When Subscript, var depends on old one as well  
            o = self.__variable(str(t_id), scope)
          t = self.__variableAssign(str(t_id), scope)
          t_found.append(str(t_id))
          if type(n) is ast.Subscript:
            self.__collect(o, func, t)
        else:
          t = self.__variable(str(t_id), scope)
        #print("target(s) ->", t)
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
        self.__collect(s, "Iter", t)
    self.generic_visit(node)

  def visit_While(self, node):
    scope = self.__scope(node)
    #self.__collect(str(node.iter),str(node.body)) 
    # print(node)
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
